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

### TERMINAL ###
# os.system('cls')

# dirname = os.getcwd()
# dirname = dirname.replace("/models/boosting", "")
# sys.path.insert(1, os.path.join(dirname, "general_classes"))
# from DataPreparation import DataPreparation
# from Performance import Performance

### FUNCTIONS ###

def normalize(X):
        return (X - X.min())/(X.max() - X.min())

### MAIN ###

def main():
    
    # Import data from train.csv, in order (Y_train, X_train, X_val, Y_val)

    DataPrep = DataPreparation('./data/train.csv', numpy_bool=True, gender=False)

    # # Create input sets of X, output 'Lead' as Y and normalize
    
    X, Y = DataPrep.raw()

    X = X.drop(columns=['Year', 'Mean Age Female',  'Mean Age Male', 'Gross'])

    Y = Y.replace("Female", -1)
    Y = Y.replace("Male", 1)
    
    # Define dataset
    # DataPrep = DataPreparation('./data/train.csv', drop_cols = [], numpy_bool = True, gender = False)
    # X, X_val, Y, Y_val = DataPrep.get_sets()
    # X = np.concatenate((X_train_b, X_val))
    # Y = np.concatenate((Y_train_b, Y_val))

    # Use SMOTE
    
    # X_res, Y_res = DataPrep.SMOTE(X, Y, num = None, perc = 300, k = 5, SMOTE_feature = -1, raw=True)
    # X_res2, Y_res2 = DataPrep.SMOTE(X, Y, num = None, perc = 200, k = 5, SMOTE_feature = 1, raw=True)

    # print(np.shape(X), np.shape(X_res))
    # X = np.concatenate((X, X_res))
    # X = np.concatenate((X, X_res2))

    # Y = np.concatenate((Y, Y_res))
    # Y = np.concatenate((Y, Y_res2))


    # Normalize input values

    normalize_X = normalize(X)
    
    # Split the data sets into 70% training data and 30% validation data

    X_train, X_val, Y_train, Y_val = skl_ms.train_test_split(normalize_X, Y, test_size = 0.3)

    # Use SMOTE to fix the inbalanced classes

    counter = Counter(Y_train)
    print(counter)
    oversample = SMOTE()
    X_train, Y_train = oversample.fit_resample(X_train, Y_train)
    counter = Counter(Y_train)
    print(counter)

    ### Simple implementation when k = 2 ###

    model = skl_nb.KNeighborsClassifier(n_neighbors=2).fit(X_train, Y_train)
    y_pred = model.predict(X_val)

    print(f'Model accuracy for k = 2: {np.mean(y_pred != Y_val):.2f}')

    ### EVALUATE AND TUNE THE MODEL ###

    ### Evaluate for which k has the least error, without and with cross-validation to see difference ###

    # Implement k-NN algorithm for different k - values

    k = 50
    K = np.arange(1, k) 

    misclassification = []

    for k in K: 
        model_loop = skl_nb.KNeighborsClassifier(n_neighbors=k).fit(X_train, Y_train)
        Y_pred_loop = model_loop.predict(X_val)
        misclassification.append(np.mean(Y_pred_loop != Y_val))

    ### Evaluate the results for different split sets and take the average ###

    n = 20

    misclassification_average = np.zeros((n, len(K)))
    
    # Create new training and validation set for each loop to take the average of it

    for i in range(n):

        X_train_loop, X_val_loop, Y_train_loop, Y_val_loop = skl_ms.train_test_split(normalize_X, Y, test_size = 0.3)

        for j, k in enumerate(K):
            model_loop = skl_nb.KNeighborsClassifier(n_neighbors=k).fit(X_train_loop, Y_train_loop)
            Y_pred_loop = model_loop.predict(X_val_loop)
            misclassification_average[i, j] = np.mean(Y_pred_loop != Y_val_loop)

    misclassification_average = np.mean(misclassification_average, axis=0)

    ### Evalute the results using cross-validation ###

    # For which n gives the best representation of the model?

    model = skl_nb.KNeighborsClassifier(n_neighbors=10)

    for i in range(2, 14):
        scores = skl_ms.cross_val_score(model, normalize_X, Y, cv=i)
        print(f'Model accuracy for n = {i}: ', np.mean(scores))

    # --> n > 7 is stabalized --> set n = 10

    n_folds = 9

    cross_val = skl_ms.KFold(n_splits = n_folds, shuffle= True)

    misclass_cross_val = np.zeros(len(K))

    # Create a loop where the cross validation changes order and calculate the error

    for index_train, index_val in cross_val.split(X):                       # cross_val.split() gives the indices for the training and validation data
        X_train_loop, X_val_loop = normalize_X.iloc[index_train], normalize_X.iloc[index_val]
        Y_train_loop, Y_val_loop = Y[index_train], Y[index_val]

        for j, k in enumerate(K):
            model_loop = skl_nb.KNeighborsClassifier(n_neighbors = k).fit(X_train_loop, Y_train_loop)
            Y_pred_loop = model_loop.predict(X_val_loop)
            misclass_cross_val[j] += np.mean(Y_pred_loop != Y_val_loop)

    misclass_cross_val /= n_folds


    # Plot the results

    plt.figure(1)
    plt.plot(K, misclassification)
    plt.title('Training error using k-NN for different k')
    plt.xlabel('k')
    plt.ylabel('Error')

    plt.figure(2)
    plt.plot(K, misclassification_average)
    plt.title(f'Average ({n} diff. splits) error using k-NN for different k')
    plt.xlabel('k')
    plt.ylabel('Error')

    plt.figure(3)
    plt.plot(K, misclass_cross_val)
    plt.title(f'Error for k-NN with, n = {n_folds} folds cross validation with different k-values')
    plt.ylabel('Error')
    plt.xlabel('k')
    plt.show()

    ### Tuning with GridSearch ###

    # Hyperparameters that could be useful to tune
    leaf_size = list(range(1, 20))
    n_neigh = list(range(5, 20))
    metric = ['minkowski', 'euclidean', 'manhattan']
    weights = ['uniform', 'distance']

    # hyperpara = dict(leaf_size = leaf_size, n_neighbors=n_neigh, weights = weights)

    # gs = skl_ms.GridSearchCV(skl_nb.KNeighborsClassifier(), hyperpara, cv=9, verbose=1, n_jobs=-1)
    
    # best_model = gs.fit(X_train, Y_train)       # -> {'leaf_size': 1, 'metric': 'euclidean', 'n_neighbors': 7, 'weights': 'uniform'}

    # print(best_model.best_params_)
    # print(best_model.best_score_)

    ### Evaluate the tuned model with cross-validation ##

    tuned_model = skl_nb.KNeighborsClassifier(leaf_size=1, n_neighbors=7, weights='uniform', metric='euclidean').fit(X_train, Y_train)

    Y_pred_tuned = tuned_model.predict(X_val)

    cross_val_score = skl_ms.cross_val_score(tuned_model, normalize_X, Y, cv=9)

    print(f'Tuned model accuracy: ', np.mean(cross_val_score))

    # Precision, recall, f1-score performance of the fitted model.
    print(skl_me.classification_report(Y_val, Y_pred_tuned))

    # Using ROC score to check our performance of the fitted model. 
    print(skl_me.roc_auc_score(Y_val, Y_pred_tuned))


### RUN (when in folder SML-LEAD-ANALYSIS) ###

if __name__ == '__main__':
    main()
