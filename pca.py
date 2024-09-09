import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Step 1: Load the dataset
dataframe = pd.read_csv("heart.csv")

# Step 2: Identify numeric columns and remove outliers using Z-Score
numeric_cols = dataframe.select_dtypes(include={'float64', 'int64'}).columns
z_scores = dataframe[numeric_cols].apply(zscore)

# Remove outliers where Z-score > 3 or Z-score < -3
no_outliers = dataframe[(z_scores < 3).all(axis=1) & (z_scores > -3).all(axis=1)]

# Step 3: Label Encoding for binary categorical columns
label_enc_columns = ['Sex', 'ExerciseAngina']
label_encoder = LabelEncoder()
for col in label_enc_columns:
    no_outliers[col] = label_encoder.fit_transform(no_outliers[col])

# Step 4: One-Hot Encoding for categorical columns with more than two categories
one_hot_enc_columns = ['ChestPainType', 'RestingECG', 'ST_Slope']
no_outliers = pd.get_dummies(no_outliers, columns=one_hot_enc_columns)

# Step 5: Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(no_outliers.drop('HeartDisease', axis=1))
scaled_df = pd.DataFrame(scaled_features, columns=no_outliers.drop('HeartDisease', axis=1).columns)

# Step 6: Prepare the data for modeling
X = scaled_df
y = no_outliers['HeartDisease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Build classification models and evaluate their accuracy
# Initialize models
models = {
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier()
}

# Train and evaluate models
accuracies = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[model_name] = accuracy

print("Accuracies without PCA:", accuracies)

# Step 8: Apply PCA for dimensionality reduction
pca = PCA(n_components=5)  # Reducing to 5 principal components for simplicity
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Retrain the models with PCA-transformed data
pca_accuracies = {}
for model_name, model in models.items():
    model.fit(X_train_pca, y_train)
    y_pred_pca = model.predict(X_test_pca)
    accuracy_pca = accuracy_score(y_test, y_pred_pca)
    pca_accuracies[model_name] = accuracy_pca

print("Accuracies with PCA:", pca_accuracies)


