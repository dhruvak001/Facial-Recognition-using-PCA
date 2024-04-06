import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Task-1: Data Preprocessing

# Load LFW dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# EDA
# Displaying the shape of the data
print("Data shape:", lfw_people.images.shape)

# Displaying the number of classes and the corresponding class names
print("Number of classes:", lfw_people.target_names.shape[0])
print("Class names:", lfw_people.target_names)

# Displaying a few sample images
fig, ax = plt.subplots(2, 5, figsize=(15, 7),
                       subplot_kw={'xticks': (), 'yticks': ()})
for i, axi in enumerate(ax.flat):
    axi.imshow(lfw_people.images[i], cmap='gray')
    axi.set_title(f"Person: {lfw_people.target_names[lfw_people.target[i]]}")
plt.show()

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(lfw_people.data, lfw_people.target, test_size=0.2, random_state=42)

# Task-2: Eigenfaces Implementation

# Implementing PCA
n_components = 150  # Chosen value for n_components
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

# EDA
# Explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Task-3: Model Training

# Choosing a classifier (K-Nearest Neighbors)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(pca.transform(X_train), y_train)

# Task-4: Model Evaluation

# Making predictions on testing data
y_pred = knn.predict(pca.transform(X_test))

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# EDA
# Visualizing a subset of Eigenfaces
eigenfaces = pca.components_.reshape((n_components, lfw_people.images.shape[1], lfw_people.images.shape[2]))

plt.figure(figsize=(15, 7))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(eigenfaces[i], cmap='gray')
    plt.title(f"Eigenface {i + 1}")
plt.show()

# Task-5: Experiment with different values of n_components in PCA and observe the impact on the performance metrics (accuracy).
for n_components in [10, 50, 100, 150, 200]:
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    knn = KNeighborsClassifier()
    knn.fit(X_train_pca, y_train)
    y_pred = knn.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'n_components: {n_components}, Accuracy: {accuracy}')
