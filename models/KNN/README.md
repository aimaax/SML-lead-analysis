Comment k-NN.

Minimize the input features because k-NN is a lazy model and is sensitive for too many inputs. Data with category words, age and amount were correlated separatly
so it could be minimized using percentage and difference between correlated features. This to maximize the impact for every data and to minimize the inputs. 
Year and Gross were removed from the data because it didn't affect the prediction. -> Noise.

k-NN is a distance based method and therefore it is important to normalize the inputs. 
Normalization methods to choose from:
    StandardScaler, MinMaxScaler, RobustScaler.
        MinMaxScaler performed best. Values from [0, 1].

Tried to use SMOTE to balance the data. SMOTE creates new data points for the unbalanced data with a slightly error. Model evaluation with cross validation makes the data points 
similiar and therefore it gets overfitted. k-NN is a too simple and lazy algorithm to use SMOTE.
    Male      785
    Female    254


Evaluation of the model with 10 fold cross validation to obtain a more valid overall result. 
    Performance of the model were evaluated with:
        Accuracy, Precision, Recall, F1, ROC and Cohen kappa score

'Hyperparameters' that were tuned:
    k-nearest neighbors, weighted data points, the distance metrics.

Model without tuning, only with k = 2, default on everything else. -> weighted = uniform, metric = minkowski
    Accuracy score: 0.8642643764002986
    Precision score: 0.9350436093565369
    Recall score: 0.881019141579763
    F1 score: 0.9060510416407508
    ROC score: 0.852558871125186
    Cohen kappa score: 0.6567899609957957

Using GridSearch to tune the model with the preprocessing data. 
    Tuned hyperparameter:
        {'metric': 'manhattan', 'n_neighbors': 8, 'weights': 'uniform'}

Model after tuning:
    Accuracy score: 0.9047050037341299
    Precision score: 0.9078974231111345
    Recall score: 0.9725512098012208
    F1 score: 0.9382914946359016
    ROC score: 0.8377006339321303
    Cohen kappa score: 0.719158617381791



