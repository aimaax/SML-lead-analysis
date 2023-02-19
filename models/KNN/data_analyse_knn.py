import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def correlation(data, threshhold):
    col_corr = set()
    corr_matrix = data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshhold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

data = pd.read_csv('./data/train.csv')

print(data['Lead'].value_counts())

print(np.shape(data))

# data["Lead age diff"] = data["Age Lead"] - data["Age Co-Lead"]
        
# # this selection was dine for the same reasons as above
# data["Mean age diff"] = data["Mean Age Male"] - data["Mean Age Female"]

# # logically, the amount of words features were going to be colinear, as seen by the
# # VIF-factors, thus theyre combined into three different features
# # (the fractions for lead and male have VIFs of ~<7, which is quite bad but
# # since some sources say VIFs<10 are acceptable and since we dont want to discard too
# # much data, theyre accepted as is)
# data["Fraction words female"] = data["Number words female"]/data["Total words"]
# data["Fraction words male"] = data["Number words male"]/data["Total words"]
# data["Fraction words lead"] = data["Number of words lead"]/data["Total words"]

# # if this turns out to increase k-fold accuracy, it stays
# data["Actor amount diff"] = data["Number of male actors"] - data["Number of female actors"]

# # the feature Year is omitted entirely partly due to it being multicolinear with 
# # features and partyl since it seems to have no large impact on the classification
# data = data.drop(["Age Lead", "Age Co-Lead", "Mean Age Male", "Mean Age Female", "Total words",
#                             "Number words female", "Number words male", "Number of words lead",
#                             "Number of male actors", "Number of female actors"],axis=1)

data['Diff frac male and female'] = data["Number words male"]/data["Total words"] - data["Number words female"]/data["Total words"]
data["Fraction words lead"] = data["Number of words lead"]/data["Total words"]
data['Fraction diff words lead and co-lead'] = data['Difference in words lead and co-lead']/data['Total words']
data["Mean age diff"] = data["Mean Age Male"] - data["Mean Age Female"]
data['Diff lead and co_lead'] = data['Age Lead'] - data['Age Co-Lead']
# data["Mean age diff"] = data["Mean Age Male"] - data["Mean Age Female"]
data['Diff number actors'] = data['Number of male actors'] - data['Number of female actors']

data = data.drop(columns=['Number words female', 'Number words male', 'Total words', 'Difference in words lead and co-lead',
                        'Mean Age Male', 'Mean Age Female', 'Age Lead', 'Age Co-Lead', 'Number of male actors', 'Number of female actors', 'Number of words lead'])

data = data.drop(columns=['Year', 'Gross'])
print(np.shape(data))

X = data.drop(columns='Lead')
Y = data['Lead']

corr_features = correlation(data, 0.7)
print(corr_features)

""" Analyse the correlation between inputs """

plt.figure(1)
sns.heatmap(X.corr(), annot=True, cmap='RdPu')
plt.title('Correlation between features')
plt.xticks(rotation=45)

plt.show()

