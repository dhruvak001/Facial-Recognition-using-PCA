# Facial Recognition using Eigenfaces


The code implements facial recognition using eigenfaces, a popular technique in computer vision and pattern recognition. It preprocesses the data, implements principal component analysis (PCA) for feature extraction, trains a K-nearest neighbors (KNN) classifier, and evaluates the model's performance using accuracy metrics. Additionally, it explores the impact of different values of n_components in PCA on model performance.

## Dataset
https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html

## Topics:

- Data Loading and Preprocessing
- Exploratory Data Analysis (EDA)
- Eigenfaces Implementation using PCA
- Model Training with K-Nearest Neighbors (KNN)
- Model Evaluation and Accuracy Calculation
- Visualization of Eigenfaces
- Experimentation with n_components in PCA
## Note:

- The code provides insights into each step with informative print statements and visualizations.
- PCA is used for dimensionality reduction, capturing the most significant features (eigenfaces) of the dataset.
- KNN classifier is trained on the reduced feature space obtained from PCA.
- Accuracy metrics help in assessing the model's performance under different scenarios.
- Experimentation with n_components allows understanding the trade-off between dimensionality reduction and classification accuracy.




