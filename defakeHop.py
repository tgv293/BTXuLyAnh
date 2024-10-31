import os
import shutil
import pickle
import numpy as np
import multiprocessing
import torch
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from multi_cwSaab import MultiChannelWiseSaab


class DefakeHop:
    def __init__(
        self,
        num_hop=3,
        kernel_sizes=[3, 3, 3],
        split_thr=0.01,
        keep_thr=0.001,
        max_channels=[10, 10, 10],
        spatial_components=[0.95, 0.95, 0.95],
        n_jobs=4,
        verbose=True,
        device="cuda:0",  # Add device parameter
    ):
        self.num_hop = num_hop
        self.kernel_sizes = kernel_sizes
        self.split_thr = split_thr
        self.keep_thr = keep_thr
        self.max_channels = max_channels
        self.spatial_components = spatial_components
        self.multi_cwSaab = None
        self.spatial_PCA = {}
        self.channel_wise_clf = {}
        self.features = {}
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.device = device

    def to_device(self, data):
        """Helper method to move data to specified device"""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.device)
        return data

    def to_numpy(self, data):
        """Helper method to convert data back to numpy"""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return data

    def fit(self, images, labels):
        if self.verbose:
            print("===============DefakeHop Training===============")

        # Move data to device
        images = self.to_device(images)
        labels = self.to_device(labels)

        self.fit_multi_cwSaab(images)
        saab_features = self.transform_multi_cwSaab(images)
        del images

        if self.verbose:
            print("===============Spatial Dimension Reduction===============")

        for hop in range(1, self.num_hop + 1):
            self.features[hop] = {}
            features = saab_features["Hop" + str(hop)]

            if len(features) > self.max_channels[hop - 1]:
                features = features[:, :, :, : self.max_channels[hop - 1]]

            # Move features to device
            features = self.to_device(features)

            self.fit_spatial_PCA(features, hop)

            if self.verbose:
                print(
                    "Input shape:",
                    features.shape[1:3],
                    features.shape[1] * features.shape[2],
                )

            for channel in range(features.shape[-1]):
                channel_wise_features = features[:, :, :, channel]
                channel_wise_features = self.transform_spatial_PCA(
                    channel_wise_features, hop
                )
                self.features[hop][channel] = channel_wise_features

                if channel == 0 and self.verbose:
                    print("Output shape:", channel_wise_features.shape[-1])

        del saab_features

        if self.verbose:
            print("===============Soft Classifiers===============")

        # Convert features and labels back to numpy for XGBoost
        features = {
            k: {c: self.to_numpy(v) for c, v in h.items()}
            for k, h in self.features.items()
        }
        labels = self.to_numpy(labels)

        fit_all_channel_wise_clf(
            features, labels, n_jobs=self.n_jobs, device=self.device
        )
        self.set_all_channel_wise_clf()
        features = self.predict_all_channel_wise_clf(self.features)

        if self.verbose:
            print("Output shape:", features.shape)

        self.features = {}
        return features

    def predict(self, images):
        if self.verbose:
            print("===============DefakeHop Prediction===============")

        # Move input to device
        images = self.to_device(images)

        saab_features = self.transform_multi_cwSaab(images)
        del images

        if self.verbose:
            print("===============Spatial Dimension Reduction===============")

        for hop in range(1, self.num_hop + 1):
            self.features[hop] = {}
            features = saab_features["Hop" + str(hop)]

            if len(features) > self.max_channels[hop - 1]:
                features = features[:, :, :, : self.max_channels[hop - 1]]

            features = self.to_device(features)

            if self.verbose:
                print(
                    "Input shape:",
                    features.shape[1:3],
                    features.shape[1] * features.shape[2],
                )

            for channel in range(features.shape[-1]):
                channel_wise_features = features[:, :, :, channel].reshape(
                    features.shape[0], -1
                )
                channel_wise_features = self.transform_spatial_PCA(
                    channel_wise_features, hop
                )
                self.features[hop][channel] = channel_wise_features

                if channel == 0 and self.verbose:
                    print("Output shape:", channel_wise_features.shape[-1])

        del saab_features

        if self.verbose:
            print("===============Soft Classifiers===============")

        # Convert features back to numpy for prediction
        features = {
            k: {c: self.to_numpy(v) for c, v in h.items()}
            for k, h in self.features.items()
        }

        features = self.predict_all_channel_wise_clf(features)

        if self.verbose:
            print("Output shape:", features.shape)

        self.features = {}
        return features

    def fit_multi_cwSaab(self, images):
        # extract features
        multi_cwSaab = MultiChannelWiseSaab(
            num_hop=self.num_hop,
            kernel_sizes=self.kernel_sizes,
            split_thr=self.split_thr,
            keep_thr=self.keep_thr,
        )
        multi_cwSaab.fit(images, verbose=self.verbose)
        self.multi_cwSaab = multi_cwSaab

    def transform_multi_cwSaab(self, images):
        return self.multi_cwSaab.transform(images, verbose=self.verbose)

    def fit_spatial_PCA(self, features, hop):
        # Check the type and shape of features before processing
        print("Features type:", type(features))
        print(
            "Features shape:",
            features.shape if isinstance(features, np.ndarray) else "Not a numpy array",
        )

        # Convert features to a numpy array if it's not already
        if not isinstance(features, np.ndarray):
            features = np.array(features)

        # Ensure features has enough dimensions for np.moveaxis
        if features.ndim < 4:
            raise ValueError(
                "Expected features to have at least 4 dimensions, but got {}".format(
                    features.ndim
                )
            )

        # Now you can safely use np.moveaxis
        features = np.moveaxis(features, -1, 1)
        features = features.reshape(features.shape[0] * features.shape[1], -1)

        pca = PCA(n_components=self.spatial_components[hop - 1], svd_solver="full")
        pca.fit(features)
        self.spatial_PCA[hop] = pca

    def transform_spatial_PCA(self, features, hop):
        # transform channel-wise data
        # flatten
        features = features.reshape(features.shape[0], -1)
        # spatial pca transformation
        pca = self.spatial_PCA[hop]
        return pca.transform(features)

    def set_all_channel_wise_clf(self):
        for hop in range(1, self.num_hop + 1):
            self.channel_wise_clf[hop] = {}
            for channel in range(len(self.features[hop])):
                clf = pickle.load(
                    open("tmp/" + str(hop) + "/" + str(channel) + ".pkl", "rb")
                )
                self.channel_wise_clf[hop][channel] = clf
        shutil.rmtree("tmp")

    def predict_all_channel_wise_clf(self, features):
        prob = []
        for hop in range(1, self.num_hop + 1):
            for channel in range(len(self.features[hop])):
                cw_prob = self.predict_channel_wise_clf(
                    self.features[hop][channel], hop, channel
                )
                prob.append(cw_prob)
        prob = np.array(prob)
        return prob.T

    def predict_channel_wise_clf(self, features, hop, channel):
        clf = self.channel_wise_clf[hop][channel]
        # Ensure features are on CPU before feeding to XGBoost
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        return clf.predict_proba(features)[:, 1]


def fit_all_channel_wise_clf(features, labels, n_jobs=4, device="cuda:0"):
    parameters = []
    for hop in range(1, len(features) + 1):
        for channel in range(len(features[hop])):
            parameters.append([features[hop][channel], labels, hop, channel, device])

    with multiprocessing.Pool(n_jobs) as pool:
        pool.starmap(fit_channel_wise_clf, parameters)


def fit_channel_wise_clf(features, labels, hop, channel, device="cuda:0"):
    print(f"===Hop {hop} Channel {channel} Start===")
    labels = labels.astype(int)

    # Create XGBoost classifier with proper device settings
    clf = XGBClassifier(
        max_depth=1,
        tree_method="hist",  # Use histogram-based algorithm
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=(len(labels[labels == 0]) / len(labels[labels == 1])),
    )

    # Ensure features are on CPU before feeding to XGBoost
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()

    clf.fit(features, labels)

    os.makedirs(f"tmp/{hop}/{channel}", exist_ok=True)
    pickle.dump(clf, open(f"tmp/{hop}/{channel}.pkl", "wb"))
    print(f"===Hop {hop} Channel {channel} Finish===")


if __name__ == "__main__":
    import time
    from sklearn.datasets import fetch_olivetti_faces

    faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True)
    data = faces.reshape(-1, 64, 64, 1)
    labels = np.ones(len(data))
    labels[: int(len(labels) / 2)] = 0
    defakehop = DefakeHop()
    prob1 = defakehop.fit(data, labels)
    prob2 = defakehop.predict(data)
    print(np.sum(np.abs(prob1 - prob2)))
