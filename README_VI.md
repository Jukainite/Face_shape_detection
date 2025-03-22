# Nhận diện Hình dạng Khuôn mặt
<a href="https://colab.research.google.com/drive/1xLL78hwNCxJR1fsIBSfLCCQAg1IFmkCw?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Mở trên Colab"></a>

<a href="https://github.com/Jukainite/Face_shape_detection/tree/main"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="Liên kết dự án trên GitHub"></a>

<a href="README.md"><img src="https://img.shields.io/badge/Translate-English-blue" alt="Dịch sang tiếng Anh"></a>

## Mục tiêu dự án
Dự án này nhằm phát triển hàm `detect_face_shape(image_path)` để dự đoán hình dạng khuôn mặt đầu tiên được phát hiện trong một ảnh đầu vào.

Dự án sử dụng **MediaPipe** để phát hiện các điểm đặc trưng trên khuôn mặt và huấn luyện mô hình **Random Forest** để phân loại hình dạng khuôn mặt dựa trên các đặc trưng đã trích xuất.

## Tập dữ liệu
Tập dữ liệu được sử dụng từ Kaggle: [Face Shape Dataset](https://www.kaggle.com/datasets/niten19/face-shape-dataset).

Tập dữ liệu bao gồm năm hình dạng khuôn mặt khác nhau:
- Hình trái tim
- Hình chữ nhật dài
- Hình oval
- Hình tròn
- Hình vuông

Phân chia dữ liệu:
- 800 ảnh/lớp để huấn luyện
- 200 ảnh/lớp để kiểm tra
- Tổng cộng: **5000 ảnh**

## Trích xuất đặc trưng
Dự án sử dụng **MediaPipe** để phát hiện **Face Mesh**, trích xuất các đặc trưng chính trên khuôn mặt:

### Các đặc trưng chính:
1. **Face Rectangularity**: Tỉ lệ diện tích khuôn mặt so với hình chữ nhật bao quanh.
2. **Middle Face Rectangularity**: Tỉ lệ diện tích phần giữa khuôn mặt so với hình chữ nhật bao quanh.
3. **Forehead Rectangularity**: Tỉ lệ diện tích trán so với hình chữ nhật bao quanh.
4. **Chin Angle**: Góc giữa cằm trái, giữa cằm và cằm phải.
5. **RBot (Lower face width / Middle face width)**: Tỉ lệ chiều rộng phần dưới khuôn mặt so với phần giữa khuôn mặt.
6. **RTop (Forehead width / Middle face width)**: Tỉ lệ chiều rộng trán so với phần giữa khuôn mặt.
7. **RTop - RBot**: Chênh lệch giữa hai tỉ lệ trên.
8. **fAR (Face width / Face height)**: Tỉ lệ chiều rộng khuôn mặt so với chiều cao khuôn mặt.
9. **Left Cheek Width**: Khoảng cách từ má trái đến mũi trái.
10. **Right Cheek Width**: Khoảng cách từ má phải đến mũi phải.
11. **Right Cheek Angle**: Góc giữa các điểm đặc trưng liên quan đến má phải.
12. **Left Cheek Angle**: Góc giữa các điểm đặc trưng liên quan đến má trái.
13. **Face Length**: Khoảng cách từ trán đến cằm.
14. **Cheekbone Width**: Khoảng cách giữa hai gò má.
15. **Jawline Width**: Chiều rộng đường xương hàm.
16. **Top Jaw Width**: Chiều rộng phần trên của xương hàm.
17. **Forehead Width (Eyebrow-based)**: Chiều rộng trán dựa trên khoảng cách giữa hai lông mày.
18. **Chin Width**: Chiều rộng cằm.
19. **Forehead width**: Chiều rộng trán

## Huấn luyện mô hình
Mô hình **Random Forest Classifier** được huấn luyện trên các đặc trưng đã trích xuất, đạt kết quả:

### Kết quả đánh giá:
```
=== Random Forest Classifier ===
Accuracy: 0.46

Classification Report:
              precision    recall  f1-score   support

       Heart       0.42      0.49      0.45       200
      Oblong       0.47      0.47      0.47       200
        Oval       0.38      0.30      0.33       200
       Round       0.46      0.36      0.40       200
      Square       0.54      0.67      0.60       200

    accuracy                           0.46      1000
   macro avg       0.45      0.46      0.45      1000
weighted avg       0.45      0.46      0.45      1000
```

## Nhận xét
Mô hình **Random Forest** không đạt hiệu suất tốt (Độ chính xác = 46%). Một số lý do có thể là:
- Các đặc trưng trích xuất chưa đủ để phân biệt rõ các hình dạng khuôn mặt.
- Tập dữ liệu có thể chứa ảnh khó phân loại.
- Random Forest có thể không phải là lựa chọn tốt nhất cho bài toán này.

## Giải pháp thay thế
Tôi đã thử nghiệm **CNN - EfficientNet B4** (được huấn luyện trước) và tinh chỉnh trên tập dữ liệu.

### Kết quả mô hình CNN:
- **Độ chính xác**: 86.1%

## Kết luận
🔥 Mô hình CNN có hiệu suất vượt trội so với Random Forest.
📌 Recall có thể cải thiện nếu cần thiết.
📌 Nếu thử nghiệm thực tế và giữ nguyên được độ chính xác, mô hình này có thể triển khai.

## Hướng dẫn sử dụng

### A. Để huấn luyện mô hình Random Forest, làm theo các bước:
1. Chuẩn bị dữ liệu.
2. Chạy `Create_data.py` để tiền xử lý và trích xuất đặc trưng.
3. Mở và chạy `Models_Training.ipynb` để huấn luyện mô hình.

### B. Để sử dụng mô hình đã huấn luyện:
```python
from Face_Shape_Detect_CNN import detect_face_shape
```
HOẶC
```python
from Face_Shape_Detect_RF import detect_face_shape
```

