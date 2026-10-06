from sklearn import datasets 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.io import arff
from pathlib import Path
import pandas as pd


#dataset: https://archive.ics.uci.edu/dataset/379/website+phishing

data_path = Path(__file__).resolve().parent / "PhishingData.arff"
arff_file = arff.loadarff(data_path)


data = pd.DataFrame(arff_file[0]).apply(pd.to_numeric)

x = data.iloc[:, : -1]
y = data.iloc[:, -1:].squeeze("columns")

#report dataset statistics
print("Number of instances:", x.shape[0])
print("Number of attributes:", x.shape[1])
print("Class distribution:")
print(y.value_counts())

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, stratify=y, random_state=42)

parameters = {
            "max_depth": [None,3,4,5,6,7,8],
            "min_samples_split": [2,3,4,5,6,8,10]
            }


tcl = tree.DecisionTreeClassifier(criterion="entropy", random_state=42)
grid = GridSearchCV(tcl, parameters, cv=5, scoring="accuracy")

tcl.fit(x_train, y_train)
grid.fit(x_train, y_train)

y_pred_base = tcl.predict(x_test)
y_pred_grid = grid.predict(x_test)

best_model = grid.best_estimator_

print("Accuracy score of baseline model:", accuracy_score(y_test, y_pred_base))
print("Accuracy score of tuned model:", accuracy_score(y_test, y_pred_grid))

print("The best maximum depth of the tree sis", grid.best_params_["max_depth"], 
      'and the best minimum number of samples required to split a node is', grid.best_params_["min_samples_split"]) 

importance = pd.Series (
    best_model.feature_importances_, 
    index=x.columns
).sort_values(ascending=False)

#find the most important attribute
most_important = importance.idxmax()

print("The most important attribute is", most_important, "with an importance score of", importance.max())

#plot the decision tree
fig = plt.figure(figsize=(30,20))
_ = tree.plot_tree(best_model, filled=True, fontsize=12, feature_names=x.columns)

