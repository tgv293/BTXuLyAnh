# DefakeHop: Bộ Phát Hiện Deepfake Hiệu Suất Cao và Nhẹ

Đây là phiên bản chính thức bằng Python của công trình nghiên cứu "DefakeHop: Bộ Phát Hiện Deepfake Hiệu Suất Cao và Nhẹ", được chấp nhận tại ICME 2021.

## Giới Thiệu

Các phương pháp phát hiện Deepfake hiện đại thường dựa trên các mạng nơ-ron sâu. Trong nghiên cứu này, chúng tôi đã đề xuất một phương pháp không sử dụng học sâu để phát hiện video Deepfake, áp dụng nguyên lý học không gian liên tiếp (Successive Subspace Learning - SSL) nhằm trích xuất các đặc trưng từ các phần khác nhau của hình ảnh khuôn mặt. Các đặc trưng này cũng được tinh lọc thêm thông qua mô-đun tinh lọc đặc trưng của chúng tôi để tạo ra một đại diện cô đọng cho khuôn mặt thật và giả.

## Khung Làm Việc

### Yêu Cầu Cài Đặt

Cài đặt các gói cần thiết bằng lệnh sau:

```bash
conda install -c anaconda pandas 
conda install -c conda-forge OpenCV
conda install -c conda-forge gst-plugins-bad=1.24.6
conda install -c anaconda scikit-image
conda install -c conda-forge matplotlib
conda install -c conda-forge scikit-learn
```

Do chúng tôi sử dụng GPU để tăng tốc các quy trình, vui lòng cài đặt xgboost bằng pip:

```bash
conda install -c conda-forge xgboost 
```

Cài đặt PyTorch:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Bạn có thể tham khảo thêm thông tin về việc cài đặt (chọn phiên bản phù hợp) tại [trang chính thức của PyTorch](https://pytorch.org/get-started/locally/).

### Dữ Liệu (https://github.com/yuezunli/celeb-deepfakeforensics)

Vui lòng sắp xếp video của bạn vào các thư mục theo cấu trúc sau:

```
train/
    real/
    fake/
test/
    real/
    fake/
```

### Tiền Xử Lý

1. Trích xuất các điểm mốc khuôn mặt bằng OpenFace. Vui lòng tham khảo [đường dẫn này](https://github.com/TadasBaltrusaitis/OpenFace) để biết thêm chi tiết.
   ```bash
   python landmark_extractor.py
   ```

2. Căn chỉnh khuôn mặt và cắt các vùng khuôn mặt:
   ```bash
   python patch_extractor.py
   ```

3. Nhận dữ liệu huấn luyện và kiểm tra:
   ```bash
   python data.py
   ```

### Cách Chạy

Chúng tôi sử dụng tập dữ liệu UADFV làm ví dụ để hướng dẫn bạn cách sử dụng mã của chúng tôi để huấn luyện và kiểm tra mô hình.

```bash
python model.py
```

Khi huấn luyện mô hình, chúng tôi sử dụng ba đối tượng sau:

- **Hình ảnh**: mảng numpy 4 chiều (N, H, W, C).
- **Nhãn**: mảng numpy 1 chiều, trong đó 1 là Deepfake và 0 là thật.
- **Tên**: mảng numpy 1 chiều lưu trữ tên các khung hình.

Tên khung hình nên theo định dạng: `{tên_video}_{số_frame}`.

**Ví dụ**: `real/0047_0786.bmp` cho biết đây là khung hình thứ 786 từ video `real/0047.mp4`.
