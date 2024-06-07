import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from empath import Empath
from scipy.sparse import hstack
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd
from scipy.sparse import csr_matrix
import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import argparse
from sklearn.metrics import f1_score
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import spacy

# Load SpaCy English tokenizer and stopwords


# Load datasets
def load_csv(filename):
    data = pd.read_csv(filename)
    return data


# Remove punctuations
def remove_punctuations(text):
    # Define regular expression pattern to match punctuations
    pattern = r'[^\w\s]'
    # Replace punctuations with empty string
    return re.sub(pattern, '', text)


# Remove stopwords for baseline model
def remove_stopwords(text):
    # Tokenize the input text
    words = nltk.word_tokenize(text)

    # Get English stopwords from NLTK corpus
    english_stopwords = set(stopwords.words('english'))

    # Remove stopwords from the tokenized words
    filtered_words = [word for word in words if word.lower() not in english_stopwords]

    # Join the filtered words back into a single string
    filtered_text = ' '.join(filtered_words)

    return filtered_text


# Remove stopwords using SpaCy
def remove_stopwords_spacy(text):
    print("running spacy tokenizer")
    # Tokenize the input text using SpaCy
    nlp = spacy.load("en_core_web_sm")
    words = nlp(text)

    # Remove stopwords from the tokenized words
    filtered_words = [token.text for token in words if not token.is_stop]

    # Join the filtered words back into a single string
    filtered_text = ' '.join(filtered_words)

    return filtered_text

# Remove emojis
def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def merge_datasets(post_data, crowd_data, test=False):
    merged_data = pd.merge(post_data, crowd_data, on='user_id')
    if test:
        selected_columns = ['post_title', 'user_id', 'subreddit', 'post_body', 'raw_label']
    else:
        selected_columns = ['post_title', 'user_id', 'subreddit', 'post_body', 'label']
    for item in selected_columns:
        merged_data[item] = merged_data[item].astype(str).str.lower()
    return merged_data


def combine_cols(data):
    data['combined_text'] = (data['post_title'].astype(str) + " " +
                                            data['subreddit'].astype(str) + " " +
                                            data['post_body'].astype(str))
    data['combined_text'] = data['combined_text'].apply(remove_punctuations)
    return data['combined_text']

# count the word "person" for future process, will remove the data entries with more than 30 person in the combined post column.
def count_occurrences(text, word):
    return text.count(word)
def data_process(main_data, label_data, train=True):
    if train:
        merged_data = merge_datasets(main_data, label_data, test=False)
    else:
        merged_data = merge_datasets(main_data, label_data, test=True)
    # use the smallest dataset for simplification purpose
    # data_for_prediction = merged_data[merged_data['subreddit']=='suicidewatch'].copy()
    data = merged_data.copy()
    data['combined_text'] = combine_cols(data)
    # remove punctuations
    data['combined_text'] = data['combined_text'].apply(remove_punctuations)
    # remove stopwords function includes nltk tokenizer, no need to flag nltk tokenizer
    # data['combined_text'] = data['combined_text'].apply(remove_stopwords_spacy)
    data['combined_text'] = data['combined_text'].apply(remove_stopwords)
    data['combined_text'] = data['combined_text'].apply(remove_emojis)

    # count the word "person"
    data['person_count'] = data['combined_text'].apply(count_occurrences, word='person')

    # Filter out entries with more than 10 occurrences of "PERSON"
    data = data[data['person_count'] <= 30]

    # Drop the 'person_count' column
    data = data.drop(columns=['person_count'])
    # drop the rows with empty label
    data['combined_text'] = data['combined_text'].str.lower()
    if train:
        data = data[data['label'] != "nan"]
        print('length of training set:', data.shape[0])
        print('empty label: ', data['label'].isna().sum())
    else:
        data = data[data['raw_label'] != "nan"]
        print('length of test set:', data.shape[0])
        print('empty label: ', data['raw_label'].isna().sum())
    # data_for_prediction.to_csv('data_for_prediction.csv',  index=False)

    # data_for_prediction = data_for_prediction.groupby('user_id').agg({'combined_text': ' '.join, 'label': 'first'}).reset_index()
    # print('length of user aggregated dataset:', data_for_prediction.shape[0])
    return data


# Mapping string label 'a' 'b' 'c' 'd' to numeric lebel 0 1 2 3
def label_mapping(data, y_pred):

    test_label = data
    test_label['prediction'] = y_pred
    test_label.index = range(0, len(test_label))
    mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    test_label['raw_label'] = test_label['raw_label'].map(mapping)
    test_label['prediction'] = test_label['prediction'].map(mapping)
    # print(test_label)
    return test_label


def user_aggregation(test_label, baseline=True):
    # make sure using baseline=False parameter if it's not baseline model.
    # baseline model which is native bayes can not predict label 'b'.
    user_true_label = pd.DataFrame(test_label.groupby('user_id').agg({'raw_label': 'first'}))
    ser_predictions = pd.DataFrame(test_label.groupby('user_id')['prediction'].value_counts().unstack(fill_value=0))
    if baseline:
        most_frequent_column = pd.DataFrame(ser_predictions[[0, 2, 3]].idxmax(axis=1))
    else:
        most_frequent_column = pd.DataFrame(ser_predictions[[0, 1, 2, 3]].idxmax(axis=1))
    merged_df = user_true_label.merge(most_frequent_column, left_index=True, right_index=True)
    merged_df.rename(columns={0: 'prediction'}, inplace=True)
    # Filter the DataFrame based on the condition where 'label' is equal to 'most_frequent_prediction'
    correctly_predicted_df = merged_df[merged_df['raw_label'] == merged_df['prediction']]
    # Get the list of correctly predicted user IDs
    correctly_predicted_user = len(correctly_predicted_df.index.tolist())

    # Print the list of correctly predicted user IDs and the number of correctly predicted user IDs
    print("Correctly Predicted User ratio:", correctly_predicted_user / merged_df.shape[0])
    print("Number of Correctly Predicted User:", correctly_predicted_user)

    true_labels = merged_df['raw_label']
    predicted_labels = merged_df['prediction']
    # Calculate the F1 score
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    # Print the F1 score
    print("User level F1 Score:", f1)


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

def run_proj(main_data, crowd_source_data, crowd_test, post_test, runmodels):

    crowd_data = load_csv(crowd_source_data)
    post_data = load_csv(main_data)
    crowd_test = load_csv(crowd_test)
    post_test = load_csv(post_test)
    # process training set
    data_for_prediction = data_process(post_data, crowd_data)

    # process test set
    data_for_test = data_process(post_test, crowd_test, train=False)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    print("Tfidfvecorizer working....")
    # Fit the vectorizer on the combined text corpus and transform the text data into TF-IDF matrix
    X_train = tfidf_vectorizer.fit_transform(data_for_prediction['combined_text'])
    y_train = data_for_prediction['label']
    X_test = tfidf_vectorizer.transform(data_for_test['combined_text'])
    y_test = data_for_test['raw_label']


    if runmodels:

        # print("Tfidfvecorizer working....")
        # tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 2))
        # # Fit the vectorizer on the combined text corpus and transform the text data into TF-IDF matrix
        # X_train = tfidf_vectorizer.fit_transform(data_for_prediction['combined_text'])
        # y_train = data_for_prediction['label']
        # X_test = tfidf_vectorizer.transform(data_for_test['combined_text'])
        # y_test = data_for_test['raw_label']
    # else:
    #     raise ValueError("Invalid vectorizer type. ")
    #     # Train the models with default hyperparameters

        print("start training")
        nb_model = MultinomialNB()
        nb_model.fit(X_train, y_train)
        print('start predicting')
        # Evaluate the models
        y_pred_nb = nb_model.predict(X_test)

        print("Naive Bayes Classifier evaluation on post level")
        print("Accuracy:", accuracy_score(y_test, y_pred_nb))
        print(classification_report(y_test, y_pred_nb))
        print('Baseline model Native Bayes evaluation on user level')
        # mapping the string labels to numeric label
        test_label = label_mapping(data_for_test, y_pred_nb)
        # aggregate test data by user_id and compute accuracy and f1 score
        user_aggregation(test_label) # make sure using baseline=False parameter if it's not baseline model.
                                     # baseline model which is native bayes can not predict label 'b'.
        print('\n---------------------------------------------------------------------------')
        print('SVM prediction evaluation on user level')
        svm_pred = pd.read_csv('./svm_predictions.csv')
        mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        svm_pred['raw_label'] = svm_pred['raw_label'].map(mapping)
        svm_pred['prediction'] = svm_pred['Predictions'].map(mapping)

        user_aggregation(svm_pred, baseline=False)
        print('SVM evaluation based on post level')
        print("Accuracy:", accuracy_score(svm_pred['raw_label'], svm_pred['prediction']))
        print(classification_report(svm_pred['raw_label'], svm_pred['prediction']))

        print('\n---------------------------------------------------------------------------')
        print('Logistic Regression prediction evaluation on user level')
        lr_pred = pd.read_csv('./logistic_regression_predictions.csv')
        mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        lr_pred['raw_label'] = lr_pred['raw_label'].map(mapping)
        lr_pred['prediction'] = lr_pred['Predictions'].map(mapping)

        user_aggregation(lr_pred, baseline=False)
        print('Logistic Regression evaluation based on post level')
        print("Accuracy:", accuracy_score(lr_pred['raw_label'], lr_pred['prediction']))
        print(classification_report(lr_pred['raw_label'], lr_pred['prediction']))





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Text Classification')
    parser.add_argument('--crowd_source_data', default="./crowd_train.csv",       action='store',      help="crowd source file")
    parser.add_argument('--main_data',         default="./shared_task_posts.csv", action='store',      help="shared task file")
    parser.add_argument('--crowd_test',        default="./crowd_test.csv", action='store', help="crowd source test file")
    parser.add_argument('--post_test',         default="./shared_task_posts_test.csv", action='store', help="shared task test file")
    parser.add_argument('--runmodels',             default=False,                     action='store_true', help='True for using tfidf as vectorizer')
    parser.add_argument('--svm',               default=False,                     action='store_true', help='True for running svm')

    args = parser.parse_args()

    run_proj(args.main_data,
           args.crowd_source_data,
           args.crowd_test,
           args.post_test,

           args.runmodels)
