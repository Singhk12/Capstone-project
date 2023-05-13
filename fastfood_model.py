import sklearn
from sklearn.utils import shuffle
from sklearn import datasets
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

import numpy
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

fast_food_nutrition_data = pd.read_csv("fastfood.csv")
predict = "restaurant"

le = preprocessing.LabelEncoder()
restaurant = le.fit_transform(list(fast_food_nutrition_data["restaurant"]))
item = le.fit_transform(list(fast_food_nutrition_data["item"]))
calories = le.fit_transform(list(fast_food_nutrition_data["calories"]))
cal_fat = le.fit_transform(list(fast_food_nutrition_data["cal_fat"]))
total_fat = le.fit_transform(list(fast_food_nutrition_data["total_fat"]))
sat_fat = le.fit_transform(list(fast_food_nutrition_data["sat_fat"]))
trans_fat = le.fit_transform(list(fast_food_nutrition_data["trans_fat"]))
cholesterol = le.fit_transform(list(fast_food_nutrition_data["cholesterol"]))
sodium = le.fit_transform(list(fast_food_nutrition_data["sodium"]))
total_carb = le.fit_transform(list(fast_food_nutrition_data["total_carb"]))
fiber = le.fit_transform(list(fast_food_nutrition_data["fiber"]))
sugar = le.fit_transform(list(fast_food_nutrition_data["sugar"]))
protein = le.fit_transform(list(fast_food_nutrition_data["protein"]))
vit_a = le.fit_transform(list(fast_food_nutrition_data["vit_a"]))
vit_c = le.fit_transform(list(fast_food_nutrition_data["vit_c"]))
calcium = le.fit_transform(list(fast_food_nutrition_data["calcium"]))
salad = le.fit_transform(list(fast_food_nutrition_data["salad"]))


x = list(zip(item, salad, calories, cal_fat, total_fat, sat_fat, trans_fat, cholesterol, sodium, total_carb, fiber, sugar, protein, vit_a, vit_c, calcium))
y = list(zip(restaurant))

x = np.array(x)
y = np.array(y)
# Test options and evaluation metric
num_folds = 5
seed = 7
scoring = 'accuracy'

# Model Test/Train
# Splitting what we are trying to predict into 4 different arrays -
# X train is a section of the x array(attributes) and vise versa for Y(features)
# The test data will test the accuracy of the model created
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=seed)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

#size of train and test subsets after splitting
np.shape(x_train), np.shape(x_test)

num_folds = 5
seed = 7

models = []
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))

# evaluate each model in turn
results = []
names = []

print("Performance on Training set")

for name, model in models:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    msg += '\n'
    print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
dt = DecisionTreeClassifier()
nb = GaussianNB()
gb = GradientBoostingClassifier()
rf = RandomForestClassifier()
best_model = rf
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
print("Best Model Accuracy Score on Test Set:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

num_classes = len(np.unique(y_train))
y_onehot_test = np.zeros((len(y_test), num_classes))
y_onehot_test[np.arange(len(y_test)), y_test] = 1
class_id = 2
class_of_interest = "Taco Bell"
RocCurveDisplay.from_predictions(y_onehot_test[:, class_id], y_pred, name=f"{class_of_interest} vs the rest", color="darkorange")
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\n Taco Bell vs (all the others)")
plt.legend()
plt.show()

for x in range(len(y_pred)):
    data_str = ", ".join(str(val) for val in x_test[x])
    print("Predicted:", y_pred[x], "Actual:", y_test[x], "Data:", "[", data_str, "]")


