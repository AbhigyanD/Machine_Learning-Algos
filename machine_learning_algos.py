# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import (
    make_regression, make_classification, make_blobs,
    load_iris, fetch_california_housing # Added for larger datasets
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


# --- 8. K-Means Clustering (Larger Synthetic Blobs Dataset) ---
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


# --- 9. Principal Component Analysis (PCA) (Larger Synthetic Data) ---
print("\n--- 9. Principal Component Analysis (PCA) ---")
# Objective: Reduce the dimensionality of the dataset while retaining most of its variance.
# Definition: Dimensionality reduction technique that transforms data to a new lower-dimensional space.
# Generate high-dimensional data with more samples
X_high_dim, _ = make_classification(n_samples=500, n_features=20, n_informative=10, n_redundant=0, random_state=42)

# Create and apply PCA
pca = PCA(n_components=5) # Reduce to 5 principal components
X_pca = pca.fit_transform(X_high_dim)

print(f"Original data shape: {X_high_dim.shape}")
print(f"Reduced data shape (5 components): {X_pca.shape}")
print(f"Explained variance ratio of components: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2f}")


# --- 10. Artificial Neural Network (ANN) / Multilayer Perceptron (MLP) (with Iris Dataset) ---
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
mlp_model.fit(X_train_scaled_mlp, y_train_clf_iris, epochs=50, batch_size=16, verbose=0) # Increased epochs for better learning

# Evaluate the model
loss_mlp, accuracy_mlp = mlp_model.evaluate(X_test_scaled_mlp, y_test_clf_iris, verbose=0)
print(f"MLP Classifier Accuracy on Iris: {accuracy_mlp:.2f}")


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
cnn_model.fit(X_train_img, y_train_img, epochs=10, batch_size=32, verbose=0)

# Evaluate the model
loss_cnn, accuracy_cnn = cnn_model.evaluate(X_test_img, y_test_img, verbose=0)
print(f"CNN Classifier Accuracy (synthetic images): {accuracy_cnn:.2f}")


# --- 12. Recurrent Neural Network (RNN) / Long Short-Term Memory (LSTM) (Synthetic Sequence Data) ---
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
lstm_model.fit(X_train_seq, y_train_seq, epochs=20, batch_size=8, verbose=0) # Increased epochs and smaller batch size

# Evaluate the model
loss_lstm, accuracy_lstm = lstm_model.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"LSTM Classifier Accuracy (synthetic text): {accuracy_lstm:.2f}")

print("\n--- All ML Algorithm Implementations Complete ---")