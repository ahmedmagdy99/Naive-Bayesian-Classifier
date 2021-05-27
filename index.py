import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#function get the intersection between two lists (the AUB)
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

data = pd.read_csv("car_data.csv", header=None)

dict = {}

predict = data.iloc[:,len(data.columns)-1]
features = data.iloc[:,0:len(data.columns)-1]

#Split the data 75% train and 25% test randomly
train_features,test_features, train_prediction, test_prediction = train_test_split(features, predict, test_size=0.25, random_state=1)

for i in train_features.columns:
    dict[i] = data[i].unique()

#unique values of labels(unacc,acc,good,vgood)
predict_unique = train_prediction.unique()
probabilities = {}
#probability of unique values of labels(unacc,acc,good,vgood)
prop_predict = train_prediction.iloc[:].value_counts().values

#Calculate the probabilities
for key, feature in dict.items():
    probabilities[key] = {}
    for value in feature:
        probabilities[key][value] = {}
        for index,label in enumerate(predict_unique):
            #Calculate the probability of every feature and add it to probabilities dictionary
            probabilities[key][value][label] = len(intersection(train_features.index[train_features.iloc[:, key] == value].tolist(),train_prediction.index[train_prediction.iloc[:] == label].tolist()))/prop_predict[index]

#Print the dictionary of values
# for key, value in probabilities.items():
#     print(f"{key}: {value}")

#Tesining Phase
predict_output = {}

product = 1
product_labels = []
final_output = []

for l,row in test_features.iterrows():
    for index_prop_predict,label in enumerate(predict_unique):
        #Multiply the probability of each feature
        for index,value in enumerate(row):
            product *= probabilities[index][value][label]
        #Multiply the probabilities by the probability of the label itself (p(b)*p(a|b)*p(c|b))
        product = product*prop_predict[index_prop_predict]
        #Add the probability of this label for this row
        product_labels.append(product)
        product = 1
    #Get the maximum probability in all labels probabilities and add its name in the list
    final_output.append(predict_unique[product_labels.index(max(product_labels))])
    product_labels.clear()
#Convert the list to series
final_output_series = pd.Series(final_output)

#Accuracy Calculation
acc = np.sum(np.equal(test_prediction.values,final_output_series.values))/len(test_prediction.values)
print("Naive Bayesian Model Accuracy = ",acc*100,'%')