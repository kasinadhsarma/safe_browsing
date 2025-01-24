import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Load the dataset (you'll need to create or obtain this dataset)
data = pd.read_csv('website_content.csv')

# Prepare the data
X = data['content']
y = data['label']  # 0 for safe, 1 for unsafe

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train and evaluate models

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized, y_train)
nb_predictions = nb_model.predict(X_test_vectorized)
print("Naive Bayes Results:")
print(classification_report(y_test, nb_predictions))

# Support Vector Machine
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vectorized, y_train)
svm_predictions = svm_model.predict(X_test_vectorized)
print("SVM Results:")
print(classification_report(y_test, svm_predictions))

# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_vectorized, y_train)
knn_predictions = knn_model.predict(X_test_vectorized)
print("KNN Results:")
print(classification_report(y_test, knn_predictions))

# Save the best performing model (you'll need to determine which one performs best)
import joblib
joblib.dump(nb_model, 'text_classification_model.joblib')
joblib.dump(vectorizer, 'text_vectorizer.joblib')

