# Cassava-Leaf-Disease-Classification
This project focuses on detecting and classifying cassava leaf diseases using ML. The EfficientNetB0 model is trained on a dataset with five classes of diseases to help farmers improve crop yield through early detection.

---

## Dataset
The dataset contains images of cassava leaves classified into five categories:
1. Healthy
2. Cassava Mosaic Disease
3. Cassava Bacterial Blight
4. Cassava Brown Streak Disease
5. Cassava Green Mite Damage

Source: [Cassava Leaf Disease Dataset on Kaggle](https://www.kaggle.com/c/cassava-leaf-disease-classification)

---

## Pipeline
1. **Data Exploration**: Visualized class distribution and sample images.
2. **Preprocessing**: Resized images to 256x256, applied augmentation, and normalized pixel values.
3. **Model Training**: Used pre-trained EfficientNetB0 with dropout and class weighting.
4. **Evaluation**: Analyzed confusion matrix and classification metrics.
5. **Predictions**: Deployed the model to classify unseen images.

---
## Results
- **Accuracy**: ~85% on validation data.
- **Precision, Recall, F1-score**:
