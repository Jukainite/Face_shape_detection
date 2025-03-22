# Face Shape Detection
<a href="https://colab.research.google.com/drive/1xLL78hwNCxJR1fsIBSfLCCQAg1IFmkCw?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Colab Trial"></a>

<a href="https://github.com/Jukainite/Face_shape_detection/tree/main"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="Github Project Link"></a>


<a href="#" onclick="translateToVietnamese()"><img src="https://img.shields.io/badge/Translate-Vietnamese-green" alt="Translate to Vietnamese"></a>

<script>
function translateToVietnamese() {
    var elements = document.querySelectorAll("body *");
    elements.forEach(element => {
        element.innerHTML = element.innerHTML
            .replace(/Project Objective/g, "Mục tiêu dự án")
            .replace(/Dataset/g, "Tập dữ liệu")
            .replace(/Feature Extraction/g, "Trích xuất đặc trưng")
            .replace(/Model Training/g, "Huấn luyện mô hình")
            .replace(/Evaluation Results/g, "Kết quả đánh giá")
            .replace(/Observations/g, "Nhận xét")
            .replace(/Alternative Solution/g, "Giải pháp thay thế")
            .replace(/Performance Review/g, "Đánh giá hiệu suất")
            .replace(/Conclusion/g, "Kết luận")
            .replace(/Usage/g, "Hướng dẫn sử dụng")
            .replace(/To train the Random Forest model, follow these steps/g, "Để huấn luyện mô hình Random Forest, hãy làm theo các bước sau")
            .replace(/Prepare Your Data/g, "Chuẩn bị dữ liệu của bạn")
            .replace(/Run `Create_data.py` to preprocess and extract features./g, "Chạy `Create_data.py` để tiền xử lý và trích xuất đặc trưng.")
            .replace(/Open and execute `Models_Training.ipynb` to train the Random Forest model./g, "Mở và chạy `Models_Training.ipynb` để huấn luyện mô hình Random Forest.")
            .replace(/To use the trained model/g, "Để sử dụng mô hình đã huấn luyện")
            .replace(/Import the function from the `.py` file from one of the models/g, "Nhập hàm từ tệp `.py` của một trong các mô hình")
            .replace(/Give the function an image path as input. You will get a string of the face shape/g, "Cung cấp đường dẫn ảnh làm đầu vào cho hàm. Bạn sẽ nhận được một chuỗi mô tả hình dạng khuôn mặt")
            .replace(/Input/g, "Đầu vào")
            .replace(/Output/g, "Đầu ra")
            .replace(/Thank you for your interest in this project!/g, "Cảm ơn bạn đã quan tâm đến dự án này!");
    });
}
</script>
## Project Objective
This project aims to develop a function `detect_face_shape(image_path)` that can predict the face shape of the first detected face in an input image.

The project utilizes **MediaPipe** to detect facial landmarks and trains a **Random Forest** model to classify face shapes based on extracted landmark features.

## Dataset
The dataset used is from Kaggle: [Face Shape Dataset](https://www.kaggle.com/datasets/niten19/face-shape-dataset).

The dataset contains five different face shapes:
- Heart
- Oblong
- Oval
- Round
- Square

Data split:
- 800 images/class for training
- 200 images/class for testing
- Total: **5000 images**

## Feature Extraction
The project uses **MediaPipe** to detect **Face Mesh**, extracting key facial features:

### Key Features:
1. **Face Rectangularity**: Ratio of face area to the bounding rectangle area.
2. **Middle Face Rectangularity**: Ratio of middle face area to the bounding rectangle area.
3. **Forehead Rectangularity**: Ratio of forehead area to the bounding rectangle area.
4. **Chin Angle**: Angle between left chin, mid chin, and right chin.
5. **RBot (Lower face width / Middle face width)**: Ratio of lower face width to middle face width.
6. **RTop (Forehead width / Middle face width)**: Ratio of forehead width to middle face width.
7. **RTop - RBot**: Difference between the two ratios.
8. **fAR (Face width / Face height)**: Ratio of face width to face height.
9. **Left Cheek Width**: Distance from left cheek to left nose.
10. **Right Cheek Width**: Distance from right cheek to right nose.
11. **Right Cheek Angle**: Angle between facial landmarks related to the right cheek.
12. **Left Cheek Angle**: Angle between facial landmarks related to the left cheek.
13. **Face Length**: Distance from forehead to chin.
14. **Cheekbone Width**: Distance between cheekbones.
15. **Jawline Width**: Jawline width.
16. **Top Jaw Width**: Upper jaw width.
17. **Forehead Width (Eyebrow-based)**: Forehead width based on the distance between eyebrows.
18. **Chin Width**: Chin width.

## Model Training
A **Random Forest Classifier** was trained on the extracted features, yielding the following results:

### Evaluation Results:
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

## Observations
The **Random Forest** model did not perform very well (Accuracy = 46%). Some possible reasons include:
- The extracted features may not be sufficient to clearly distinguish face shapes.
- The dataset may contain images that are difficult to classify.
- Random Forest may not be the best choice for this problem.
- ----> I may need more time to enhance this issue

## Alternative Solution
I experimented with **CNN - EfficientNet B4** (pretrained) and fine-tuned it on the dataset. Detailed training can be found here: [Kaggle Notebook](https://www.kaggle.com/code/phamkhacduy/test-shape-detection)

### CNN Model Results:
- **Accuracy**: 86.1%
- **Precision**: 85.58%
- **Recall**: 82.60%
- **F1-score**: 83.87%

### Performance Review:
✅ **High Precision (~85.58%)** → Model makes fewer incorrect predictions.

🟡 **Moderate Recall (~82.60%)** → Some faces may still be misclassified.

✅ **Good F1-score (~83.87%)** → Balanced between Precision and Recall.

## Conclusion
🔥 The CNN model significantly outperforms Random Forest.
📌 Recall can be improved if necessary.
📌 If real-world testing maintains accuracy, this model is suitable for deployment.

---

## Usage

### A. To train the Random Forest model, follow these steps:
#### 1. Prepare Your Data
Ensure your dataset is organized in a directory structure where each subdirectory corresponds to a face shape category. For example:
train/
   heart/
       image1.jpg
       image2.jpg
       ...
   oval/
       image1.jpg
       image2.jpg
       ...
   round/
       image1.jpg
       image2.jpg
       ...
   square/
       image1.jpg
       image2.jpg
       ...
#### 2. Run `Create_data.py` to preprocess and extract features.
You will need to run this file twice. Once with train folder and once with the test one. Remember to set the config in the code
#### 3. Open and execute `Models_Training.ipynb` to train the Random Forest model.


### B. To use the trained model:
for prediction you can refer to 2 `Main` jupyter notebook file or import the built-in function `detect_face_shape` by importing from the 2 model `.py` files !!

1. Import the function from the `.py` file from one of the models
```python
from Face_Shape_Detect_CNN import detect_face_shape
```
OR

```python
from Face_Shape_Detect_RF import detect_face_shape
```

2. Give the function an image path as input. You will get a string of the face shape
```python
image_path = "path/to/your/image.jpg"
predicted_shape = detect_face_shape(image_path)
print("Predicted Face Shape:", predicted_shape)
```
- **Input**: Image file path
- **Output**: A string representing the predicted face shape (e.g., "Oval", "Round", etc.)


Thank you for your interest in this project! 🚀







