# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score, confusion_matrix # Added confusion_matrix here
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import (
    make_regression, make_classification, make_blobs,
    load_iris, fetch_california_housing
)

# Supervised Learning Algorithms
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

# Unsupervised Learning Algorithms
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Deep Learning (Neural Networks) - using TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# --- Visualization Libraries ---
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Starting ML Algorithm Implementations with Larger Datasets ---")

# --- 1. Linear Regression (with California Housing Dataset) ---
print("\n--- 1. Linear Regression ---")
# Objective: Predict median house values (continuous) based on various features.
# Definition: Models linear relationship between input features and continuous output.
# Load California Housing dataset
housing = fetch_california_housing()
X_reg = housing.data
y_reg = housing.target

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale features (important for many ML models, especially regression with varied feature scales)
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Create and train the model
linear_model = LinearRegression()
linear_model.fit(X_train_reg_scaled, y_train_reg)

# Make predictions
y_pred_reg = linear_model.predict(X_test_reg_scaled)

# Evaluate the model (Objective: Minimize Mean Squared Error)
mse_linear = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Linear Regression MSE on California Housing: {mse_linear:.2f}")

# --- Linear Regression Plotting ---
plt.figure(figsize=(12, 5))

# Plot 1: Actual vs. Predicted Values
plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
plt.scatter(y_test_reg, y_pred_reg, alpha=0.3)
plt.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], color='red', linestyle='--') # 45-degree line
plt.title('Linear Regression: Actual vs. Predicted Values')
plt.xlabel('Actual Values (Median House Value)')
plt.ylabel('Predicted Values (Median House Value)')
plt.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Residuals Plot
residuals = y_test_reg - y_pred_reg
plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
plt.scatter(y_pred_reg, residuals, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Linear Regression: Residuals Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Actual - Predicted)')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout() # Adjust layout to prevent overlap
plt.show() # Display the plots


# --- 2. Logistic Regression (with Iris Dataset) ---
print("\n--- 2. Logistic Regression ---")
# Objective: Classify Iris flower species (multi-class) based on their measurements.
# Definition: Used for classification, predicts probability using a sigmoid/softmax function.
# Load Iris dataset
iris = load_iris()
X_clf_iris = iris.data
y_clf_iris = iris.target

# Split data
X_train_clf_iris, X_test_clf_iris, y_train_clf_iris, y_test_clf_iris = train_test_split(
    X_clf_iris, y_clf_iris, test_size=0.2, random_state=42, stratify=y_clf_iris # stratify for balanced classes
)

# Scale features
scaler_lr = StandardScaler()
X_train_clf_iris_scaled = scaler_lr.fit_transform(X_train_clf_iris)
X_test_clf_iris_scaled = scaler_lr.transform(X_test_clf_iris)

# Create and train the model (multi_class='multinomial' for more than 2 classes)
logistic_model = LogisticRegression(max_iter=200, multi_class='multinomial', solver='lbfgs', random_state=42)
logistic_model.fit(X_train_clf_iris_scaled, y_train_clf_iris)

# Make predictions
y_pred_clf_lr = logistic_model.predict(X_test_clf_iris_scaled)

# Evaluate the model (Objective: Maximize Accuracy)
accuracy_lr = accuracy_score(y_test_clf_iris, y_pred_clf_lr)
print(f"Logistic Regression Accuracy on Iris: {accuracy_lr:.2f}")


# --- 3. Decision Tree (Classification with Iris Dataset) ---
print("\n--- 3. Decision Tree (Classification) ---")
# Objective: Classify Iris flower species.
# Definition: Tree-like model for decisions based on feature values.
# Using the Iris dataset (X_clf_iris, y_clf_iris)

# Create and train the model
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_clf_iris, y_train_clf_iris) # No scaling needed for tree-based models

# Make predictions
y_pred_dt = dt_classifier.predict(X_test_clf_iris)

# Evaluate the model (Objective: Maximize Accuracy)
accuracy_dt = accuracy_score(y_test_clf_iris, y_pred_dt)
print(f"Decision Tree Classifier Accuracy on Iris: {accuracy_dt:.2f}")


# --- 4. Random Forest (Classification with Iris Dataset) ---
print("\n--- 4. Random Forest (Classification) ---")
# Objective: Classify Iris flower species with improved robustness.
# Definition: Ensemble of multiple decision trees, improving accuracy and reducing overfitting.
# Using the Iris dataset (X_clf_iris, y_clf_iris)

# Create and train the model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42) # 100 trees
rf_classifier.fit(X_train_clf_iris, y_train_clf_iris)

# Make predictions
y_pred_rf = rf_classifier.predict(X_test_clf_iris)

# Evaluate the model (Objective: Maximize Accuracy)
accuracy_rf = accuracy_score(y_test_clf_iris, y_pred_rf)
print(f"Random Forest Classifier Accuracy on Iris: {accuracy_rf:.2f}")


# --- 5. Support Vector Machine (SVC for Classification with Iris Dataset) ---
print("\n--- 5. Support Vector Machine (SVC for Classification) ---")
# Objective: Classify Iris flower species by finding the optimal separation hyperplane.
# Definition: Finds an optimal hyperplane to separate classes with maximum margin.
# Using the Iris dataset (X_clf_iris, y_clf_iris)

# Scale features for SVM
scaler_svm = StandardScaler()
X_train_scaled_svm = scaler_svm.fit_transform(X_train_clf_iris)
X_test_scaled_svm = scaler_svm.transform(X_test_clf_iris)

# Create and train the model (Using RBF kernel for potentially better performance on non-linear data)
svm_classifier = SVC(kernel='rbf', random_state=42)
svm_classifier.fit(X_train_scaled_svm, y_train_clf_iris)

# Make predictions
y_pred_svm = svm_classifier.predict(X_test_scaled_svm)

# Evaluate the model (Objective: Maximize Accuracy)
accuracy_svm = accuracy_score(y_test_clf_iris, y_pred_svm)
print(f"SVM Classifier Accuracy on Iris: {accuracy_svm:.2f}")


# --- 6. K-Nearest Neighbors (KNN for Classification with Iris Dataset) ---
print("\n--- 6. K-Nearest Neighbors (KNN for Classification) ---")
# Objective: Classify Iris flower species based on closest neighbors.
# Definition: Classifies based on the majority class of its 'k' nearest neighbors.
# Using the Iris dataset (X_clf_iris, y_clf_iris)

# Scale features for KNN
scaler_knn = StandardScaler()
X_train_scaled_knn = scaler_knn.fit_transform(X_train_clf_iris)
X_test_scaled_knn = scaler_knn.transform(X_test_clf_iris)

# Create and train the model
knn_classifier = KNeighborsClassifier(n_neighbors=5) # K=5
knn_classifier.fit(X_train_scaled_knn, y_train_clf_iris)

# Make predictions
y_pred_knn = knn_classifier.predict(X_test_scaled_knn)

# Evaluate the model (Objective: Maximize Accuracy)
accuracy_knn = accuracy_score(y_test_clf_iris, y_pred_knn)
print(f"KNN Classifier Accuracy on Iris: {accuracy_knn:.2f}")


# --- 7. Naive Bayes (Gaussian Naive Bayes with Iris Dataset) ---
print("\n--- 7. Naive Bayes (Gaussian Naive Bayes) ---")
# Objective: Classify Iris flower species using a probabilistic approach.
# Definition: Probabilistic classifier based on Bayes' Theorem with feature independence assumption.
# Using the Iris dataset (X_clf_iris, y_clf_iris)

# Create and train the model
gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train_clf_iris, y_train_clf_iris)

# Make predictions
y_pred_gnb = gnb_classifier.predict(X_test_clf_iris)

# Evaluate the model (Objective: Maximize Accuracy)
accuracy_gnb = accuracy_score(y_test_clf_iris, y_pred_gnb)
print(f"Naive Bayes Classifier Accuracy on Iris: {accuracy_gnb:.2f}")


# Store accuracies for comparison (MOVED TO HERE, after all models are run)
classifier_accuracies = {
    'Logistic Regression': accuracy_lr,
    'Decision Tree': accuracy_dt,
    'Random Forest': accuracy_rf,
    'SVM': accuracy_svm,
    'K-Nearest Neighbors': accuracy_knn,
    'Naive Bayes': accuracy_gnb
}

print("\n--- Classification Model Accuracies ---")
for model, acc in classifier_accuracies.items():
    print(f"{model}: {acc:.2f}")

# --- Classification Plotting ---

# Plot 1: Confusion Matrix for Logistic Regression (Example)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
cm_lr = confusion_matrix(y_test_clf_iris, y_pred_clf_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Logistic Regression Confusion Matrix (Iris)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Plot 2: Bar Chart of Classifier Accuracies
plt.subplot(1, 2, 2)
sns.barplot(x=list(classifier_accuracies.keys()), y=list(classifier_accuracies.values()), palette='viridis')
plt.title('Accuracy Comparison of Classification Models (Iris)')
plt.ylabel('Accuracy')
plt.ylim(0.0, 1.0) # Accuracy is between 0 and 1
plt.xticks(rotation=45, ha='right') # Rotate labels for readability

plt.tight_layout()
plt.show()


# --- 8. K-Means Clustering (Larger Synthetic Blobs Dataset) ---
# Removed the duplicate section here
print("\n--- 8. K-Means Clustering ---")
# Objective: Discover natural groupings/clusters within a dataset.
# Definition: Partitions data into K clusters based on similarity (distance to centroid).
# Generate a larger synthetic dataset for clustering
X_clusters, y_true = make_blobs(n_samples=1000, centers=3, cluster_std=0.75, random_state=0)

# Create and train the model (no 'y' for unsupervised learning)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # n_init for robust initialization
kmeans.fit(X_clusters)

# Get cluster assignments
labels = kmeans.labels_

# Evaluate clustering (Objective: Find meaningful clusters, higher Silhouette Score is better)
# Note: Silhouette score requires more than 1 cluster and less than n_samples-1 clusters
if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(X_clusters) - 1:
    silhouette_avg = silhouette_score(X_clusters, labels)
    print(f"K-Means Silhouette Score (1000 samples): {silhouette_avg:.2f}")
else:
    print("Cannot calculate Silhouette Score with current cluster configuration.")
print(f"First 10 cluster labels: {labels[:10]}")

# --- K-Means Plotting ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_clusters[:, 0], y=X_clusters[:, 1], hue=labels, palette='viridis', legend='full', s=50, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, color='red', label='Centroids', edgecolors='black')
plt.title('K-Means Clustering with 3 Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- Optional: Elbow Method for K-Means (Demonstration) ---
# Sum of squared distances of samples to their closest cluster center.
wcss = []
for i in range(1, 11): # Try k from 1 to 10
    # Suppress KMeans warning about 'init' being deprecated in 1.4 for 'n_init'
    # By setting n_init explicitly, you usually avoid it.
    kmeans_elbow = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans_elbow.fit(X_clusters)
    wcss.append(kmeans_elbow.inertia_) # Inertia is the WCSS

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# --- 9. Principal Component Analysis (PCA) (Larger Synthetic Data) ---
print("\n--- 9. Principal Component Analysis (PCA) ---")
# Objective: Reduce the dimensionality of the dataset while retaining most of its variance.
# Definition: Dimensionality reduction technique that transforms data to a new lower-dimensional space.
# Generate high-dimensional data with more samples
X_high_dim, y_true_pca = make_classification(n_samples=500, n_features=20, n_informative=10, n_redundant=0, random_state=42)

# It's good practice to scale data before PCA
scaler_pca = StandardScaler()
X_high_dim_scaled = scaler_pca.fit_transform(X_high_dim)

# Create and apply PCA
pca = PCA(n_components=5) # Reduce to 5 principal components
X_pca = pca.fit_transform(X_high_dim_scaled) # Use scaled data

print(f"Original data shape: {X_high_dim.shape}")
print(f"Reduced data shape (5 components): {X_pca.shape}")
print(f"Explained variance ratio of components: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2f}")

# --- PCA Plotting ---
plt.figure(figsize=(10, 5))

# Plot 1: Explained Variance Ratio
plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.title('Explained Variance Ratio per Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Plot 2: Cumulative Explained Variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# If you were to reduce to 2 components, you could plot them:
# pca_2d = PCA(n_components=2)
# X_pca_2d = pca_2d.fit_transform(X_high_dim_scaled)
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=y_true_pca, palette='deep', alpha=0.7)
# plt.title('PCA 2 Components (Color by True Class)')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()


# --- 10. Artificial Neural Network (MLP for Classification) ---
print("\n--- 10. Artificial Neural Network (MLP for Classification) ---")
# Objective: Classify Iris flower species using a neural network.
# Definition: A basic neural network with input, hidden, and output layers.
# Using the Iris dataset (X_clf_iris, y_clf_iris)

# Scale features for ANN
scaler_mlp = StandardScaler()
X_train_scaled_mlp = scaler_mlp.fit_transform(X_train_clf_iris)
X_test_scaled_mlp = scaler_mlp.transform(X_test_clf_iris)

# Build the MLP model
# Output layer changed to Dense(3, activation='softmax') for 3 classes in Iris
mlp_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled_mlp.shape[1],)), # Input layer + 1st hidden layer
    Dense(32, activation='relu'), # 2nd hidden layer
    Dense(3, activation='softmax') # Output layer for multi-class classification (Iris has 3 classes)
])

# Compile the model
mlp_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', # Use sparse_categorical_crossentropy for integer labels
                  metrics=['accuracy'])

# Train the model
# Verbose=0 hides training output, set to 1 for progress updates
mlp_history = mlp_model.fit(X_train_scaled_mlp, y_train_clf_iris, epochs=50, batch_size=16, verbose=0) # Increased epochs for better learning

# Evaluate the model
loss_mlp, accuracy_mlp = mlp_model.evaluate(X_test_scaled_mlp, y_test_clf_iris, verbose=0)
print(f"MLP Classifier Accuracy on Iris: {accuracy_mlp:.2f}")

# --- MLP Plotting (Training History) ---
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(mlp_history.history['accuracy'])
plt.title('MLP Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left') # No validation split, so only train accuracy
plt.grid(True, linestyle='--', alpha=0.6)

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(mlp_history.history['loss'])
plt.title('MLP Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left') # No validation split, so only train loss
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


# --- 11. Convolutional Neural Network (CNN) (Synthetic Image Data) ---
print("\n--- 11. Convolutional Neural Network (CNN for Image Classification) ---")
# Objective: Classify synthetic images. For real applications, this would be image recognition.
# Definition: Specialized neural network for processing grid-like data like images.
# For real CNN applications, you would load large image datasets like MNIST, Fashion MNIST, CIFAR-10.
num_samples = 500 # Increased samples
img_rows, img_cols = 32, 32 # Larger image size
num_classes = 5 # More classes

# Create random images (values between 0 and 1)
X_img = np.random.rand(num_samples, img_rows, img_cols)
y_img = np.random.randint(0, num_classes, num_samples) # Random labels

# Reshape data for CNN input (add channel dimension)
X_img = X_img.reshape(num_samples, img_rows, img_cols, 1)

# Split data
X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
    X_img, y_img, test_size=0.2, random_state=42
)

# Build the CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(), # Flatten the 2D feature maps into a 1D vector
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax') # Output layer for multi-class classification
])

# Compile the model
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', # Use sparse_categorical_crossentropy for integer labels
                  metrics=['accuracy'])

# Train the model
cnn_history = cnn_model.fit(X_train_img, y_train_img, epochs=10, batch_size=32, verbose=0)

# Evaluate the model
loss_cnn, accuracy_cnn = cnn_model.evaluate(X_test_img, y_test_img, verbose=0)
print(f"CNN Classifier Accuracy (synthetic images): {accuracy_cnn:.2f}")

# --- CNN Plotting (Training History) ---
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(cnn_history.history['accuracy'])
plt.title('CNN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['loss'])
plt.title('CNN Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


# --- 12. Recurrent Neural Network (LSTM) (Synthetic Sequence Data) ---
print("\n--- 12. Recurrent Neural Network (LSTM for Sequence Classification) ---")
# Objective: Classify short text sequences. For real applications, this would be sentiment analysis, translation, etc.
# Definition: Handles sequential data by maintaining a hidden state/memory. LSTMs use gates to manage long-term dependencies.
# For real LSTM applications, you would use large text corpora (e.g., IMDB reviews, news articles) or time series data.
sentences = [
    "this is a very good and positive movie", "this is a totally bad and negative movie", "great film, really enjoyed it",
    "terrible acting, wasted my time", "absolutely loved the plot and characters", "hated every minute, total garbage",
    "a must see for everyone", "avoid at all costs, worst movie ever",
    "happy to recommend this one", "disappointing and boring", "excellent storyline", "very dull experience"
]
labels_seq = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]) # 1 for positive, 0 for negative

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=100, oov_token="<unk>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, maxlen=15, padding='post') # Pad to a max length of 15, post-padding

# Split data
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
    padded_sequences, labels_seq, test_size=0.25, random_state=42 # Increased test size for more test data
)

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 32 # Increased embedding dimension
max_len = 15

# Build the LSTM model
lstm_model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len), # Converts words to dense vectors
    LSTM(64), # LSTM layer with 64 units
    Dense(1, activation='sigmoid') # Output layer for binary classification
])

# Compile the model
lstm_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

# Train the model
lstm_history = lstm_model.fit(X_train_seq, y_train_seq, epochs=20, batch_size=8, verbose=0) # Increased epochs and smaller batch size

# Evaluate the model
loss_lstm, accuracy_lstm = lstm_model.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"LSTM Classifier Accuracy (synthetic text): {accuracy_lstm:.2f}")

# --- LSTM Plotting (Training History) ---
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(lstm_history.history['accuracy'])
plt.title('LSTM Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(lstm_history.history['loss'])
plt.title('LSTM Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

print("\n--- All ML Algorithm Implementations Complete ---")