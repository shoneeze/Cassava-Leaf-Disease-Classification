# Cassava-Leaf-Disease-Classification

This repository contains a deep learning approach to classify cassava leaves into five disease categories using an EfficientNet-based model.

## Business Context

Cassava is a staple crop for over 800 million people worldwide. However, diseases like Mosaic and Bacterial Blight can reduce yields by up to 40%, impacting food security and livelihoods.

This project proposes a deep learning model that:
- Identifies leaf diseases with ~74% accuracy.
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
   - Applied data augmentation (rotation, zoom, flips, brightness adjustments)
   - Normalized pixel values.
3. **Model Training**:
   - EfficientNetB0 backbone with pretrained ImageNet weights (transfer learning).
   - Added a custom classification head which consists of a global average pooling (GAP) layer and finally a fully connected (dense) layer that outputs the class predictions.
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
- The model’s training accuracy reached approximately 75%, indicating it fit the training set reasonably well.
- Training accuracy steadily improved, but validation accuracy fluctuated initially.
<img width="819" alt="Screenshot 2025-01-30 at 5 55 39 PM" src="https://github.com/user-attachments/assets/30b23a28-43bb-41f1-9e43-8bb4b3ceaa8e">

**Confusion Matrix**:
- The confusion matrix shows the model tends to predict certain classes (often the majority class) more accurately while frequently misclassifying minority classes into that same dominant class.
- Class 3 (Cassava Brown Streak Disease) had the highest F1-score (0.86).
- Class 0 (Healthy) had a high recall (0.80) but low precision (0.37), meaning many false positives.
- Class 4 (Cassava Green Mite Damage) improved with adjusted thresholds, reaching 64% recall.

<img width="819" alt="Screenshot 2025-01-30 at 5 56 37 PM" src="https://github.com/user-attachments/assets/60880c3d-b6d5-49df-844b-a72e140616c1">

**ROC Curves**:
- All classes achieved AUC > 0.90, indicating strong model performance.
- Class 3 had the highest separability (AUC = 0.96), while Class 4 was harder to distinguish (AUC = 0.90).

<img width="819" alt="Screenshot 2025-01-30 at 5 57 03 PM" src="https://github.com/user-attachments/assets/68942f3c-a523-4bd2-b4cd-d3f2696e7771">

## Next Steps
- Further fine-tune EfficientNet by unfreezing last layers for better feature extraction
- Increase dataset size or apply domain adaptation techniques to improve minority class predictions.
- Alternative Architectures: Consider a Vision Transformer or a larger EfficientNet variant (like B3)
- Use Grad-CAM to visualize which regions of the image the model is prioritizing.
