import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, cohen_kappa_score, roc_curve


class Performance():
    def __init__(self, Y_pred, Y_true, gender=False):
        self.Y_pred = Y_pred
        self.Y_true = Y_true
        self.gender = gender
    
    ##########################################################
    ## 
    
    def accuracy(self):
        return np.sum(self.Y_true == self.Y_pred) / len(self.Y_true)
    
    def balanced_accuracy(self):
        return balanced_accuracy_score(self.Y_true, self.Y_pred)
    
    def precision(self):
        male_precision = precision_score(self.Y_true, self.Y_pred, pos_label="Male" if self.gender else 1)
        female_precision = precision_score(self.Y_true, self.Y_pred, pos_label="Female" if self.gender else -1)
        
        return (male_precision, female_precision)
    
    def recall(self):
        male_recall = recall_score(self.Y_true, self.Y_pred, pos_label="Male" if self.gender else 1)
        female_recall = recall_score(self.Y_true, self.Y_pred, pos_label="Female" if self.gender else -1)
        
        return (male_recall, female_recall)
    
    def confusion(self):
        return confusion_matrix(self.Y_true, self.Y_pred)
    
    def f1(self):
        male_f1 = f1_score(self.Y_true, self.Y_pred, pos_label="Male" if self.gender else 1)
        female_f1 = f1_score(self.Y_true, self.Y_pred, pos_label="Female" if self.gender else -1)
    
        return (male_f1, female_f1)

    def cohen(self):
        return cohen_kappa_score(self.Y_true, self.Y_pred)
    
    def roc(self):
        m_fpr, m_tpr, m_thres = roc_curve(self.Y_true, self.Y_pred, pos_label="Male" if self.gender else 1)
        f_fpr, f_tpr, f_thres = roc_curve(self.Y_true, self.Y_pred, pos_label="Female" if self.gender else -1)
        
        return ((m_fpr, m_tpr, m_thres), (f_fpr, f_tpr, f_thres))
    
    ##########################################################
    ## COLLECTION
    
    def combination(self, data):
        funcs = [self.accuracy, self.balanced_accuracy, self.precision,
                 self.recall, self.f1, self.cohen]
        
        for func, key in zip(funcs, data.keys()):
            data[key].append(func())
            
    def print_combination(self, data):
        for key in data.keys():
            try:
                print(key + ":", round(sum(data[key])/len(data[key]),2))
            except ZeroDivisionError:
                print(key, "undefined due to strange parameters")  
            except:
                print(key + ":", round(sum(data[key][0])/len(data[key][0]),2),
                      round(sum(data[key][1])/len(data[key][1]),2))
    