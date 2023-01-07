# Imports, settings and first dataset view
import pandas as pd
import seaborn as sns
import numpy as np
import json

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import metrics, __all__, datasets

# Setting features for further feature extraction by choosing columns
# Some will be "simply" encoded via label encoding and others with HashingVectorizer

# On these headers we will run a "simple" BOW
SIMPLE_HEADERS = ['request.headers.Accept-Encoding',
                  'request.headers.Connection',
                  'request.headers.Host',
                  'request.headers.Accept',
                  'request.method',
                  'request.headers.Accept-Language',
                  'request.headers.Sec-Fetch-Site',
                  'request.headers.Sec-Fetch-Mode',
                  'request.headers.Sec-Fetch-Dest',
                  'request.headers.Sec-Fetch-User',
                  'response.status',
                  'url_len',
                  # 'request.headers.Sec-Ch-Ua-Platform', Uncomment this for dataset 4
                  # 'request.headers.Upgrade-Insecure-Requests', Uncomment this for dataset 4
                  # 'request.headers.Cache-Control' Uncomment this for dataset 4
                  ]

# On these headers we will run HashingVectorizer
COMPLEX_HEADERS = ['request.headers.User-Agent',
                   'request.headers.Set-Cookie',
                   'request.headers.Date',
                   # 'request.url',
                   'response.headers.Content-Type',
                   'response.body',
                   'response.headers.Location',
                   'request.headers.Content-Length',
                   'request.headers.Cookie',
                   'response.headers.Set-Cookie',
                   # 'request.headers.Sec-Ch-Ua-Mobile' Uncomment this for dataset 4
                   ]

COLUMNS_TO_REMOVE = ['request.body',
                     'response.headers.Content-Length',
                     'request.headers.Date']

# Set pandas to show all columns when you print a dataframe
pd.set_option('display.max_columns', None)

# Global setting here you choose the dataset number and classification type for the model
dataset_number = 1  # Options are [1, 2, 3, 4]
test_type = 'label'  # Options are ['label', 'attack_type']


# This is ran's code for code preprocessing:
def data_preprocessing():
    # Read the json and read it to a pandas dataframe object.
    with open(f'./dataset_{str(dataset_number)}_train.json') as file:
        raw_ds = json.load(file)
    # Turning this json into a table
    df = pd.json_normalize(raw_ds, max_level=2)

    # Fill the black attack tag lines with "Benign" string
    df['request.Attack_Tag'] = df['request.Attack_Tag'].fillna('Benign')
    df['attack_type'] = df['request.Attack_Tag']
    df['label'] = df.apply(lambda row: categorize(row), axis=1)

    # After finishing the arrangements we delete the irrelevant column
    df.drop('request.Attack_Tag', axis=1, inplace=True)

    # Remove all NAN columns or replace with desired string
    # This loop iterates over all of the column names which are all NaN
    #This is a new feature.
    df['url_len'] = df.apply(lambda x: len(x['request.url']), axis=1)
    df.drop('request.url', axis=1, inplace=True)
    for column in df.columns[df.isna().any()].tolist():
        # df.drop(column, axis=1, inplace=True)
        df[column] = df[column].fillna('None')
    df = vectorize_df(df)

    features_list = df.columns.to_list()
    features_list.remove('label')
    features_list.remove('attack_type')

    return df, features_list


# This function will be used in the lambda below to iterate over the label columns
def categorize(row):
    if row['request.Attack_Tag'] == 'Benign':
        return 'Benign'
    return 'Malware'


# columns and run some feature extraction models
def vectorize_df(df):
    le = LabelEncoder()
    h_vec = HashingVectorizer(n_features=4)

    # Run LabelEncoder on the chosen features
    for column in SIMPLE_HEADERS:
        df[column] = le.fit_transform(df[column])

    # Run HashingVectorizer on the chosen features
    for column in COMPLEX_HEADERS:
        newHVec = h_vec.fit_transform(df[column])
        df[column] = newHVec.todense()

    # Remove some columns that may be needed.. (Or not, you decide)
    for column in COLUMNS_TO_REMOVE:
        df.drop(column, axis=1, inplace=True)
    return df


def data_train_rf(df, updated_features_list):
    # We convert the feature list to a numpy array, this is required for the model fitting
    X = df[updated_features_list].to_numpy()

    # This column is the desired prediction we will train our model on
    y = np.stack(df[test_type])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1765, random_state=50, stratify=y)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # using metrics module for accuracy calculation
    #print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred)) #Uncomment for model accuracy.
    return clf, X_test, y_test


def feature_selection(df, feature_list, to_drop):
    #First we drop the feature that we want to frop from correlation.
    for drop in to_drop:
        if drop in feature_list:
            feature_list.remove(drop)
    #We train 100% of the train data set to get feature importance.
    X = df[feature_list].to_numpy()
    # This column is the desired prediction we will train our model on
    y = np.stack(df[test_type])
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)
    sort = clf.feature_importances_.argsort()
    updated_feature_list = []
    for i in sort:
        if clf.feature_importances_[i] > 0:
            updated_feature_list.append(feature_list[i])

        #print("feature: ", feature_list[i], " ", clf.feature_importances_[i], " ", i)

    return updated_feature_list


def print_result_report(clf, X_test, y_test):
    # We print our results
    sns.set(rc={'figure.figsize': (15, 8)})
    predictions = clf.predict(X_test)
    true_labels = y_test
    cf_matrix = confusion_matrix(true_labels, predictions)
    clf_report = classification_report(true_labels, predictions, digits=5)
    heatmap = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g',
                          xticklabels=np.unique(true_labels),
                          yticklabels=np.unique(true_labels))
    plt.show()
    print(clf_report)


def show_correlation(df):
    color = plt.get_cmap('RdYlGn')  # default color
    color.set_bad('lightblue')  # if the value is bad the color would be lightblue instead of white
    plt.figure(figsize=(16, 6))

    # Store heatmap object in a variable to easily access it when you want to include more features (such as title).
    # Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
    heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap=color)

    plt.show()


def run_val(clf, updated_features_list):
    with open(f'./CompetitionDatasets/dataset_{str(dataset_number)}_val.json') as file:
        raw_ds = json.load(file)
    test_df = pd.json_normalize(raw_ds, max_level=2)
    test_df['url_length'] = test_df.apply(lambda x: len(x['request.url']), axis=1)
    test_df.drop('request.url', axis=1, inplace=True)
    for column in test_df.columns[test_df.isna().any()].tolist():
        test_df[column] = test_df[column].fillna('None')
    test_df = vectorize_df(test_df)
    X = test_df[updated_features_list].to_numpy()
    predictions = clf.predict(X)
    # Save your predictions
    enc = LabelEncoder()
    np.savetxt(f'./dataset_{str(dataset_number)}_{test_type}_result.txt', enc.fit_transform(predictions), fmt='%2d')


def drop_feature(df):
    temp_df = df.copy()
    temp_df.drop(columns=["attack_type", "label"], inplace=True)

    # create correlation  matrix
    corr_matrix = temp_df.corr().abs()

    # select upper traingle of correlation matrix
    # k=1 is to skip the alachson
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find index of columns with correlation greater than 0.7
    #Ofcourse you can change this in between datasets, according to correlation heatmap.
    # Drops highly correlated features in positive and negative.
    to_drop = [column for column in upper.columns if any(upper[column] >= 0.7)]

    return to_drop


if __name__ == '__main__':
    df, feature_list = data_preprocessing()
    #show_correlation(df) Uncomment this to show correlation heatmap.
    to_drop = drop_feature(df) #which features to drop according to correlation heatmap.
    updated_feature_list = feature_selection(df, feature_list, to_drop) #which features to drop according to feature importance.
    clf, X_test, y_test = data_train_rf(df, updated_feature_list)
    # run_val(clf, updated_feature_list) #Run this to write the results of val to the file.

    print_result_report(clf, X_test, y_test) #Prints the recall and confusion matrix.
