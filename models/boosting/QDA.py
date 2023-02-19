import os
import sys
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/models/boosting", "")
sys.path.insert(1, os.path.join(dirname, "general_classes"))
from DataPreparation import DataPreparation
from Performance import Performance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

def main():
    # Set data
    path = dirname + "/data/train.csv"
    drop_cols = ["Total words","Year", "Gross"]
    DataPrep = DataPreparation(path, drop_cols = [], numpy_bool = True, gender = False, normalize = False)
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()
    X_train2 = X_train
    X_test2 = X_test
    Y_train2 = Y_train
    Y_test2 = Y_test
    
    sm2 = SMOTE(k_neighbors = 5)
    X_res_a, Y_res_a = sm2.fit_resample(X_train2, Y_train2)
    X_train2 = np.concatenate((X_train2,X_res_a))
    Y_train2 = np.concatenate((Y_train2, Y_res_a))
   
    qda2 = QuadraticDiscriminantAnalysis(reg_param = 0.0)


    model2 = qda2.fit(X_train2, Y_train2)
    y_pred2 = model2.predict(X_test2)
    print(classification_report(Y_test2,y_pred2))

    perf = Performance(y_pred2,Y_test2)
    print(f"Kappa: {perf.cohen()}")


    # Use SMOTE
    """
    X_res, Y_res = DataPrep.SMOTE(num=None,perc=300,k=5,SMOTE_feature=-1)
    X_res2, Y_res2 = DataPrep.SMOTE(num=None,perc=100,k=5,SMOTE_feature=1)

    X_train = np.concatenate((X_train, X_res))
    X_train = np.concatenate((X_train, X_res2))

    Y_train = np.concatenate((Y_train, Y_res))
    Y_train = np.concatenate((Y_train, Y_res2))
    
    """
    
    sm = SMOTE(k_neighbors = 5)
    X_res, Y_res = sm.fit_resample(X_train, Y_train)
    X_train = np.concatenate((X_train, X_res))
    Y_train = np.concatenate((Y_train, Y_res))
    
    X_train = np.concatenate((X_train, X_test))
    Y_train = np.concatenate((Y_train, Y_test))
  
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    qda = QuadraticDiscriminantAnalysis(reg_param = 0.0)
    
    """
    qda.fit(X_train, Y_train)
    y_pred = qda.predict(X_test)
    
    Perfor = Performance(y_pred, Y_test)
    accuracy = Perfor.accuracy()
    print(accuracy)
    """

    
    model = qda.fit(X_train, Y_train)



    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X_train , Y_train, scoring="accuracy", cv=cv)
    for i in range(len(scores)):
        print(f"Iteration: {i}, accuracy: \t {scores[i]}")

    print(f"\nMean accuracy: \t {np.mean(scores)}")

    params = [{'reg_param': np.linspace(0,1,100)}]
    Grids = GridSearchCV(qda, params, cv=4)
    Grids.fit(X_train, Y_train)
    reg_params = Grids.best_params_['reg_param']
    print("Best reg_param for model is " + str(reg_params))

    
    

if __name__ == "__main__":
    main()
