import numpy as np
from numpy import linalg as LA, sqrt
from skimage.util.shape import view_as_windows


class Saab:
    def __init__(self, kernel_size=3, bias_flag=False):
        self.bias = None
        self.features_mean = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.kernel_size = kernel_size
        self.bias_flag = bias_flag

    def PCA(self, X):
        self.eigenvalues, self.eigenvectors = LA.eig(np.cov(X, rowvar=0))
        self.eigenvalues = np.abs(self.eigenvalues)
        idx = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[idx][:-1]  # Remove the last eigenvalue
        self.eigenvectors = self.eigenvectors[:, idx][
            :, :-1
        ]  # Remove corresponding eigenvector
        return self

    def patch_extraction(self, images):
        N = images.shape[0]
        H = images.shape[1] - self.kernel_size + 1
        W = images.shape[2] - self.kernel_size + 1
        C = images.shape[-1]
        images = view_as_windows(images, (1, self.kernel_size, self.kernel_size, 1))
        return images.reshape(N, H, W, C * self.kernel_size**2)

    def fit(self, images, max_images=10000, max_patches=1000000, seed=777):
        images = np.array(images)
        images = images.astype("float32")  # Use float32 to reduce memory usage

        # Subsample images
        if len(images) > max_images:
            print("sampling " + str(max_images) + " images")
            np.random.seed(seed)
            images = images[np.random.choice(len(images), max_images, replace=False), :]

        N = images.shape[0]
        H = images.shape[1] - self.kernel_size + 1
        W = images.shape[2] - self.kernel_size + 1
        C = images.shape[3] * self.kernel_size**2

        # Collect patches
        patches = self.patch_extraction(images)
        del images
        if len(patches) > max_patches:
            print("sampling " + str(max_patches) + " patches")
            np.random.seed(seed)
            patches = patches[
                np.random.choice(len(patches), max_patches, replace=False), :
            ]

        # Flatten
        patches = patches.reshape(N * H * W, C)
        if self.bias_flag:
            self.bias = np.max(LA.norm(patches, axis=1))

        # Remove mean
        self.features_mean = np.mean(patches, axis=0, keepdims=True)
        patches -= self.features_mean

        # Remove patches mean
        patches_mean = np.mean(patches, axis=1, keepdims=True)
        patches -= patches_mean

        # Calculate eigenvectors and eigenvalues
        self.PCA(patches)

        return self

    def _transform_batch(self, images, n_channels=-1):
        N = images.shape[0]
        H = images.shape[1] - self.kernel_size + 1
        W = images.shape[2] - self.kernel_size + 1
        C = images.shape[3] * self.kernel_size**2

        # Create patches
        patches = self.patch_extraction(images)
        del images

        # Flatten
        patches = patches.reshape(N * H * W, C)

        # Remove mean
        patches -= self.features_mean

        # Remove patches mean
        patches_mean = np.mean(patches, axis=1, keepdims=True)
        patches -= patches_mean

        if n_channels == -1:
            kernels = self.eigenvectors
            n_channels = C - 1
        else:
            kernels = self.eigenvectors[:, :n_channels]

        if self.bias_flag:
            patches = patches + self.bias / sqrt(C)
            return np.matmul(patches, kernels).reshape(N, H, W, n_channels)
        else:
            return np.matmul(patches, kernels).reshape(N, H, W, n_channels)

    def transform(self, images, n_channels=-1, batch_size=10000):  # Reduce batch size
        images = np.array(images)
        images = images.astype("float32")  # Use float32 instead of float64

        N = images.shape[0]
        H = images.shape[1] - self.kernel_size + 1
        W = images.shape[2] - self.kernel_size + 1
        C = images.shape[3] * self.kernel_size**2
        if n_channels == -1:
            n_channels = C - 1

        output = np.zeros((N, H, W, n_channels), dtype="float32")  # Use float32
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            print("Batch", i // batch_size, "from", i, "to", end - 1)
            out = self._transform_batch(images[i:end], n_channels=n_channels)
            output[i:end] = out
            del out  # Free memory

        return output


if __name__ == "__main__":
    import time
    from sklearn.datasets import load_digits

    digits = load_digits()
    data = digits.data
    data = data.reshape(-1, 8, 8, 1)

    # Test Saab
    saab = Saab(bias_flag=True)
    start = time.time()
    saab.fit(data)
    print("training time:", time.time() - start, "s")

    start = time.time()
    output = saab.transform(data)
    print("transformation time:", time.time() - start, "s")

    # Test PCA
    data = data.reshape(-1, 64)
    from sklearn.decomposition import PCA

    pca = PCA()
    start = time.time()
    pca.fit(data)
    print("sklearn pca training time:", time.time() - start, "s")

    start = time.time()
    output = pca.transform(data)
    print("sklearn pca transform time:", time.time() - start, "s")

    saab = Saab()
    start = time.time()
    saab.PCA(data)
    print("numpy pca training time:", time.time() - start, "s")
    print(
        "difference of eigenvalues:",
        np.sum(np.abs(pca.explained_variance_ - saab.eigenvalues)),
    )

    assert np.sum(np.abs(pca.explained_variance_ - saab.eigenvalues)) <= 10**-10
    print("dot product of eigenvectors of sklearn pca and numpy pca:")
    print(np.diag(np.matmul(pca.components_, saab.eigenvectors)))
