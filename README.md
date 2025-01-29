# Cassava-Leaf-Disease-Classification

This repository contains a deep learning approach to classify cassava leaves into five disease categories using an EfficientNet-based model.

## Business Context

Cassava is a staple crop for over 800 million people worldwide. However, diseases like Mosaic and Bacterial Blight can reduce yields by up to 40%, impacting food security and livelihoods.

This project proposes a deep learning model that:
- Identifies leaf diseases with ~70–80% accuracy.
- Helps farmers take timely actions, reducing losses.

## Dataset
Source: [Cassava Leaf Disease Dataset on Kaggle](https://www.kaggle.com/c/cassava-leaf-disease-classification)

The dataset contains images of cassava leaves classified into five categories:
1. Healthy
2. Cassava Mosaic Disease
3. Cassava Bacterial Blight
4. Cassava Brown Streak Disease
5. Cassava Green Mite Damage

## Pipeline
1. **Exploratory Data Analysis**:
   - Visualized class imbalance
   - Confirmed uniform image dimensions (various ~512×512 images).
2. **Preprocessing**:
   - Resized images to 256x256
   - Applied data augmentation (rotation, zoom, flips, brightness adjustments)
   - Normalized pixel values.
3. **Model Training**:
   - EfficientNetB0 backbone with pretrained ImageNet weights (transfer learning).
   - Added a custom classification head which consists of a global average pooling (GAP) layer, followed by a dropout layer to reduce overfitting, and finally a fully connected (dense) layer that outputs the class predictions.
   - Applied class weighting to handle class imbalance.
   - Callbacks: ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau.
4. **Evaluation**:
   - Assessed training vs. validation performance across epochs.
   - Analyzed confusion matrix and ROC curves for each class.
   - Computed precision, recall, F1-score per class.
5. **Prediction**:
   - Predicted disease class for test data and saved the output as a CSV.
## Results

**Training vs. Validation Accuracy**:
- The model’s training accuracy reached approximately 70–80%, indicating it fit the training set reasonably well.
- However, validation accuracy hovered around 30–40%, revealing a significant gap and overfitting.
**Confusion Matrix**:
  
 <img width="819" alt="Screenshot 2025-01-29 at 3 28 53" src="https://github.com/user-attachments/assets/d2201cf7-cdd2-4131-8775-b3b424ddb7f0">

- The confusion matrix shows the model tends to predict certain classes (often the majority class) more accurately while frequently misclassifying minority classes into that same dominant class.
- This imbalance underscores overfitting and highlights the need for additional regularization or more balanced data.
**ROC Curves**:
Per-class ROC curves confirm that a few classes have moderate to fair discriminatory power, while others show poor separation, resulting in lower AUCs.
Overall, the model struggles to distinguish among certain classes, reinforcing the need for additional improvements.

<img width="819" alt="Screenshot 2025-01-29 at 1 31 44" src="https://github.com/user-attachments/assets/da3386e3-02e6-4b8a-a03f-bca4e3e96492">

## Next Steps
- Unfreeze More Layers: Fine-tune deeper layers of EfficientNet with a lower learning rate to better adapt to cassava leaf characteristics.
- Enhance Regularization: Increase dropout or L2 regularization to combat overfitting.
- Data Augmentation: Try advanced methods like MixUp or CutMix or refine current augmentations for more variety.
- Alternative Architectures: Consider a Vision Transformer or a larger EfficientNet variant (like B3)
