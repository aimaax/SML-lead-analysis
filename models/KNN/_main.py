### IMPORTS ###
import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter

from general_classes import *
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
import sklearn.metrics as skl_me
import sklearn.feature_selection as skl_fs

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler



### FUNCTIONS ###

def pre_proc(data):
    """ Minimize the features to exclude noice, many of them were correlated. k-NN is also sensitive for many features. """

    data['Diff frac male and female'] = data["Number words male"]/data["Total words"] - data["Number words female"]/data["Total words"]
    data["Fraction words lead"] = data["Number of words lead"]/data["Total words"]
    data['Fraction diff words lead and co-lead'] = data['Difference in words lead and co-lead']/data['Total words']
    data["Mean age diff"] = data["Mean Age Male"] - data["Mean Age Female"]
    data['Diff age lead and co-lead'] = data['Age Lead'] - data['Age Co-Lead']
    data['Diff number actors'] = data['Number of male actors'] - data['Number of female actors']

    data = data.drop(columns=['Number words female', 'Number words male', 'Total words', 'Difference in words lead and co-lead',
                            'Mean Age Male', 'Mean Age Female', 'Age Lead', 'Age Co-Lead', 'Number of male actors', 'Number of female actors', 'Number of words lead'])

    data = data.drop(columns=['Year', 'Gross'])

    return data

def scale_X(data, scaler = 1):
    if scaler == 1:
        scaler = StandardScaler()
    elif scaler == 2:
        scaler = MinMaxScaler()
    elif scaler == 3:
        scaler = RobustScaler()

    data = scaler.fit_transform(data)

    return data

def gender_num(data):
    data = data.replace('Female', -1)
    data = data.replace('Male', 1)
    return data

def sep_X_Y(data):
    Y = data['Lead']
    X = data.drop(columns=['Lead'])
    return X, Y

def train_smote(X_train, Y_train):

    X_train, Y_train = SMOTE().fit_resample(X_train, Y_train)

    return X_train, Y_train

def evaluation_cross_val(X, Y, n_folds = 10, tuned=True):

    accuracy = np.zeros(n_folds)
    precision = np.zeros(n_folds)
    recall = np.zeros(n_folds)
    F1 = np.zeros(n_folds)
    cohen_kappa = np.zeros(n_folds)
    roc = np.zeros(n_folds)
    
    cross_val = skl_ms.KFold(n_splits = n_folds, shuffle= True, random_state=False)

    for i, (index_train, index_val) in enumerate(cross_val.split(X)):     # cross_val.split() gives the indices for the training and validation data
        X_train_loop, X_val_loop = X[index_train], X[index_val]
        Y_train_loop, Y_val_loop = Y[index_train], Y[index_val]

        if tuned:
            model = skl_nb.KNeighborsClassifier(metric='manhattan', n_neighbors=8, weights='uniform').fit(X_train_loop, Y_train_loop)
        else:
            model = skl_nb.KNeighborsClassifier(n_neighbors=2).fit(X_train_loop, Y_train_loop)

        Y_pred_loop = model.predict(X_val_loop)

        accuracy[i] = skl_me.accuracy_score(Y_val_loop, Y_pred_loop)
        precision[i] = skl_me.precision_score(Y_val_loop, Y_pred_loop)
        recall[i] = skl_me.recall_score(Y_val_loop, Y_pred_loop)
        F1[i] = skl_me.f1_score(Y_val_loop, Y_pred_loop)
        cohen_kappa[i] = skl_me.cohen_kappa_score(Y_val_loop, Y_pred_loop)
        roc[i] = skl_me.roc_auc_score(Y_val_loop, Y_pred_loop)

    return np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(F1), np.mean(cohen_kappa), np.mean(roc)


def main():
    """Set random seed to easier tune the model"""

    np.random.seed(10)


    """Import data from train.csv """
    
    data = pd.read_csv('./data/train.csv')


    """ Pre-processing the data """

    data = pre_proc(data)


    """ Replace the gender string to int, Male = 1, Female = -1 """

    data = gender_num(data)


    """Split the data into inputs, X, and output 'Lead' as Y.""" 
    X, Y = sep_X_Y(data)


    """Scale features values, because k-NN is a distance-based method """

    X = scale_X(X, scaler = 2)


    """Use SMOTE to fix the inbalanced classes, don't include X_val and Y_val. Dont want model to 'see' similiar data. DID NOT WORK. OVERFITTING """

    # print(f'Male(1) & Female(-1) before smote: {Counter(Y_train)}')

    # X_train, Y_train = train_smote(X_train, Y_train)
    
    # print(f'Male(1) & Female(-1) after smote: {Counter(Y_train)}')

    # # New X and Y after smote

    # X = np.concatenate((X_train, X_val))
    # Y = np.concatenate((Y_train, Y_val))


    """ Evaluate model without tuning for k = 2. Model implemented in evaluation_cross_val function """

    unt_accuracy, unt_precision, unt_recall, unt_f1, unt_cohen_kappa, unt_roc = evaluation_cross_val(X, Y, n_folds=10, tuned=False)

    print('Untuned model:')
    print(f'Accuracy score: {unt_accuracy}')
    print(f'Precision score: {unt_precision}')
    print(f'Recall score: {unt_recall}')
    print(f'F1 score: {unt_f1}')
    print(f'ROC score: {unt_roc}')
    print(f'Cohen kappa score: {unt_cohen_kappa}')


    """  # for k in np.arange(1, 20):  # This will be tested with gridsearch
        #     model = skl_nb.KNeighborsClassifier(n_neighbors=k).fit(X_train, Y_train)
        #     cross_val_score = skl_ms.cross_val_score(model, X, Y,  scoring='accuracy', cv=10).mean()
        #     print(f'For k = {k}. Cross validation score = {cross_val_score}') """


    """ Tuning with GridSearch """

    # Hyperparameters that could be useful to tune
    n_neigh = list(range(1, 30))
    metric =  ['manhattan'] #['minkowski', 'euclidean', 'manhattan'] # Already tested all the combination. Commented away this to compute less.
    weights = ['uniform'] #['uniform', 'distance']

    hyperpara = dict(n_neighbors=n_neigh, weights = weights, metric = metric)

    gs = skl_ms.GridSearchCV(skl_nb.KNeighborsClassifier(), hyperpara, cv=10, verbose=1, n_jobs=-1)     

    tuned_n = np.zeros(10)


    for i in np.arange(10):
        X_train, _, Y_train, _ = skl_ms.train_test_split(X, Y, test_size=0.3) 
        tuned_model = gs.fit(X_train, Y_train)      # -> {'metric': 'manhattan', 'n_neighbors': 8, 'weights': 'uniform'}
        print(f'Tuned model parameters: {tuned_model.best_params_}')
        dictionary_para = tuned_model.best_params_
        tuned_n[i] += dictionary_para['n_neighbors']
    
    avg_tuned_n = np.mean(tuned_n)

    print(f'Tuned nearest neighbors: {avg_tuned_n}')
    

    """ Evaluate the results """
    
    accuracy, precision, recall, f1, cohen_kappa, roc = evaluation_cross_val(X, Y, n_folds=10, tuned=True)

    print('Tuned model:')
    print(f'Accuracy score: {accuracy}')
    print(f'Precision score: {precision}')
    print(f'Recall score: {recall}')
    print(f'F1 score: {f1}')
    print(f'ROC score: {roc}')
    print(f'Cohen kappa score: {cohen_kappa}')

    
### RUN (when in folder SML-LEAD-ANALYSIS) ###

if __name__ == '__main__':
    main()