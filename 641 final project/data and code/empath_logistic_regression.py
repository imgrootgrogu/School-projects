import pandas as pd
import numpy as np
import csv
# Load CSV data
def load_csv(filename):
    data = pd.read_csv(filename)
    return data


# Load data
merged_data = load_csv('/Users/meenamallela/Documents/UMD/MSML641_TxtClassification/data_for_prediction.csv')
merged_data2 = load_csv('/Users/meenamallela/Documents/UMD/MSML641_TxtClassification/data_for_test.csv')

# Create a copy of the DataFrame to avoid SettingWithCopyWarning
data_for_prediction = merged_data.copy()
data_for_test = merged_data2.copy()

data_for_prediction = data_for_prediction.dropna(subset=['label'])
data_for_test = data_for_test.dropna(subset=['raw_label'])      #'raw_label

import nltk
nltk.download('punkt')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
from scipy.stats import randint
from empath import Empath 
from scipy.sparse import csr_matrix


# Function to encode TF-IDF features
def encode_tfidf_features(X):
    # Get feature names from TfidfVectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out()
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
# X_train, X_test, y_train, y_test = train_test_split(data_for_prediction["combined_text"],data_for_prediction["label"],test_size=0.2, random_state=42)

# # TF-IDF Vectorization
# tfidf_vectorizer = TfidfVectorizer(ngram_range=(2,2))
# X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
# X_test_tfidf = tfidf_vectorizer.transform(X_test)


tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 2))
# Fit the vectorizer on the combined text corpus and transform the text data into TF-IDF matrix
X_train = tfidf_vectorizer.fit_transform(data_for_prediction['combined_text'])
y_train = data_for_prediction['label']
X_test = tfidf_vectorizer.transform(data_for_test['combined_text'])
y_test = data_for_test['raw_label'] #['raw_label']

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd 

# Compute cosine similarity between document pairs
# cosine_similarities = cosine_similarity(X_train_tfidf, X_train_tfidf)

# nlp = spacy.load("en_core_web_sm")

# Tokenize your text data
# tokenized_data = [word_tokenize(text.lower()) for text in data_for_prediction["combined_text"]]

# Tokenize text data for Empath scoring
tokenized_data_train = [word_tokenize(text.lower()) for text in data_for_prediction["combined_text"]]
tokenized_data_test = [word_tokenize(text.lower()) for text in data_for_test['combined_text']]

# tokenized_data_train = [word_tokenize(text.lower()) for text in X_train]
# tokenized_data_test = [word_tokenize(text.lower()) for text in X_test]

# Train Word2Vec model
# word2vec_model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)

# Save the trained model
# word2vec_model.save("word2vec_model.bin")

from empath import Empath
from scipy.sparse import hstack

# Create an Empath object
lexicon = Empath()


# Compute Empath scores for each tokenized document in train data
empath_scores_train = []
for i, (doc_tokens, tfidf_vector) in enumerate(zip(tokenized_data_train, X_train)):     #X_train
    doc_text = ' '.join(str(doc_tokens))  # Convert tokens back to text
    doc_empath_score = lexicon.analyze(doc_text, normalize=True)
    empath_scores_train.append(doc_empath_score)
    # print(f"Document {i + 1}:")
    # print("Text:", doc_text)
    # print("TF-IDF Vector:", tfidf_vector)
    # print("Empath Score:", doc_empath_score)
    # print()

# Compute Empath scores for each tokenized document in test data
empath_scores_test = []
for i, (doc_tokens, tfidf_vector) in enumerate(zip(tokenized_data_test, X_test)):       # X_test
    doc_text = ' '.join(str(doc_tokens))  # Convert tokens back to text
    doc_empath_score = lexicon.analyze(doc_text, normalize=True)
    empath_scores_test.append(doc_empath_score)

# print("got the empath scores")

# Convert Empath scores to DataFrame for both train and test data
empath_df_train = pd.DataFrame(empath_scores_train)
empath_df_test = pd.DataFrame(empath_scores_test)



# Encode TF-IDF and Empath features for train data
X_train_tfidf_encoded, tfidf_encoded_feature_names = encode_tfidf_features(X_train)   #X_train
X_train_empath_encoded, empath_encoded_feature_names = encode_empath_features(empath_df_train)

# Encode TF-IDF and Empath features for test data
X_test_tfidf_encoded, _ = encode_tfidf_features(X_test)       # X_test
X_test_empath_encoded, _ = encode_empath_features(empath_df_test)

# Concatenate encoded TF-IDF and Empath feature matrices for train data
X_train_combined_encoded = hstack([X_train_tfidf_encoded, X_train_empath_encoded])

# Concatenate encoded TF-IDF and Empath feature matrices for test data
X_test_combined_encoded = hstack([X_test_tfidf_encoded, X_test_empath_encoded])

# Concatenate encoded feature names
encoded_feature_names = tfidf_encoded_feature_names + empath_encoded_feature_names


from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
from sklearn.metrics import classification_report

# Step 3: Define Parameter Grid
param_grid = {
    'C': uniform(loc=0.1, scale=10),  # Regularization parameter
    'gamma': ['scale', 'auto'],        # Kernel coefficient
    'kernel': ['linear', 'rbf']       # Kernel type
}

# # Step 4: Random Search with Parallel Processing
# svm_model = svm.SVC()
# random_search = RandomizedSearchCV(estimator=svm_model, param_distributions=param_grid, n_iter=10, cv=5, scoring='f1_micro', random_state=42, n_jobs=-1)
# random_search.fit(X_train_combined_encoded, y_train)

# # Step 5: Fit Model
# best_svm_model = random_search.best_estimator_

# # Step 6: Evaluate Model
# y_pred = best_svm_model.predict(X_test_combined_encoded)
# report = classification_report(y_test, y_pred)

# print("Best parameters:", random_search.best_params_)
# print("Classification Report:")
# print(report)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.stats import uniform


# Define the logistic regression model
logistic = LogisticRegression(max_iter=1000)

# Define hyperparameters to search
param_distributions = {
    'C': uniform(loc=0, scale=4),  # Regularization parameter
    'penalty': ['l1', 'l2']        # Penalty (L1 or L2)
}

# Create the randomized search object
random_search = RandomizedSearchCV(estimator=logistic, 
                                   param_distributions=param_distributions, 
                                   n_iter=20,  # Number of parameter settings that are sampled
                                   scoring='f1_micro',  # Use F1 score as the scoring metric
                                   cv=5,       # Cross-validation folds
                                   random_state=42)

# Perform randomized search on the training data
random_search.fit(X_train_combined_encoded, y_train)

# Step 5: Fit Model
best_logreg_model = random_search.best_estimator_

# Step 6: Evaluate Model
y_pred = best_logreg_model.predict(X_test_combined_encoded)
predictions_df = pd.DataFrame({'Predictions': y_pred})
results_df = pd.concat([data_for_prediction['user_id'],y_test, predictions_df], axis=1)
results_df.to_csv('logistic_regression_predictions.csv', index=False)

report = classification_report(y_test, y_pred)

print("Best parameters:", random_search.best_params_)
print("Classification Report:")
print(report)