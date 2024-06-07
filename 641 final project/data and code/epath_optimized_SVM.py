import pandas as pd
import numpy as np
import csv
# Load CSV data
def load_csv(filename):
    data = pd.read_csv(filename)
    return data


# Load data
merged_data = load_csv('./data_for_prediction.csv')


# Create a copy of the DataFrame to avoid SettingWithCopyWarning
data_for_prediction = merged_data.copy()

data_for_prediction = data_for_prediction.dropna(subset=['label'])

import nltk
nltk.download('punkt')


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from empath import Empath
from scipy.sparse import hstack
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd
from scipy.sparse import csr_matrix

# Function to encode TF-IDF features
def encode_tfidf_features(X):
    # Get feature names from TfidfVectorizer
    feature_names = vectorizer.get_feature_names_out()
    # Append prefix to feature names to denote TF-IDF origin
    encoded_feature_names = [f"tfidf_{feature}" for feature in feature_names]
    # Create a CSR matrix with encoded feature names
    return csr_matrix(X), encoded_feature_names

# Function to encode Empath features
def encode_empath_features(X):
    # Get feature names from Empath scores
    feature_names = X.columns
    # Append prefix to feature names to denote Empath origin
    encoded_feature_names = [f"empath_{feature}" for feature in feature_names]
    # Convert DataFrame to CSR matrix with encoded feature names
    return csr_matrix(X.values), encoded_feature_names


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data_for_prediction["combined_text"],data_for_prediction["label"],test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(2,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Tokenize your text data
# tokenized_data = [word_tokenize(text.lower()) for text in data_for_prediction["combined_text"]]

# Tokenize text data for Empath scoring
tokenized_data_train = [word_tokenize(text.lower()) for text in X_train]
tokenized_data_test = [word_tokenize(text.lower()) for text in X_test]

# # Train Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_data_train, vector_size=100, window=5, min_count=1, workers=4)

# # Save the trained model
word2vec_model.save("word2vec_model.bin")



# Create an Empath object
lexicon = Empath()


# Compute Empath scores for each tokenized document in train data
empath_scores_train = []
for i, (doc_tokens, tfidf_vector) in enumerate(zip(tokenized_data_train, X_train_tfidf)):
    doc_text = ' '.join(doc_tokens)  # Convert tokens back to text
    doc_empath_score = lexicon.analyze(doc_text, normalize=True)
    empath_scores_train.append(doc_empath_score)
    # print(f"Document {i + 1}:")
    # print("Text:", doc_text)
    # print("TF-IDF Vector:", tfidf_vector)
    # print("Empath Score:", doc_empath_score)
    # print()

# Compute Empath scores for each tokenized document in test data
empath_scores_test = []
for i, (doc_tokens, tfidf_vector) in enumerate(zip(tokenized_data_test, X_test_tfidf)):
    doc_text = ' '.join(doc_tokens)  # Convert tokens back to text
    doc_empath_score = lexicon.analyze(doc_text, normalize=True)
    empath_scores_test.append(doc_empath_score)

# Convert Empath scores to DataFrame for both train and test data
empath_df_train = pd.DataFrame(empath_scores_train)
empath_df_test = pd.DataFrame(empath_scores_test)

# Encode TF-IDF and Empath features for train data
X_train_tfidf_encoded, tfidf_encoded_feature_names = encode_tfidf_features(X_train_tfidf)
X_train_empath_encoded, empath_encoded_feature_names = encode_empath_features(empath_df_train)

# Encode TF-IDF and Empath features for test data
X_test_tfidf_encoded, _ = encode_tfidf_features(X_test_tfidf)
X_test_empath_encoded, _ = encode_empath_features(empath_df_test)

# Concatenate encoded TF-IDF and Empath feature matrices for train data
X_train_combined_encoded = hstack([X_train_tfidf_encoded, X_train_empath_encoded])

# Concatenate encoded TF-IDF and Empath feature matrices for test data
X_test_combined_encoded = hstack([X_test_tfidf_encoded, X_test_empath_encoded])

# Concatenate encoded feature names
encoded_feature_names = tfidf_encoded_feature_names + empath_encoded_feature_names

# Print shapes of final feature matrices
print("Shape of X_train_combined_encoded:", X_train_combined_encoded.shape)
print("Shape of X_test_combined_encoded:", X_test_combined_encoded.shape)

from sklearn import svm
from sklearn.metrics import classification_report

# Define the SVM model
svm_model = svm.SVC(C=1.66, gamma='scale', kernel='linear')

# Fit the model
svm_model.fit(X_train_combined_encoded, y_train)

# Evaluate Model
y_pred = svm_model.predict(X_test_combined_encoded)
report = classification_report(y_test, y_pred)
predictions_df = pd.DataFrame({'Predictions': y_pred})
results_df = pd.concat([data_for_prediction['user_id'],y_test, predictions_df], axis=1)
results_df.to_csv('svm_predictions.csv', index=False)

print("SVM Model with Manual Parameter Settings:")
print("C:", svm_model.C)
print("gamma:", svm_model.gamma)
print("kernel:", svm_model.kernel)
print("\nClassification Report:")
print(report)