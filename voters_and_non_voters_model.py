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


voters_and_non_voters = pd.read_csv("nonvoters_dataset.csv")
predict = "voters"

le = preprocessing.LabelEncoder()
RespId = le.fit_transform(list(voters_and_non_voters["RespId"]))
gender = le.fit_transform(list(voters_and_non_voters["gender"]))
race = le.fit_transform(list(voters_and_non_voters["race"]))
educ = le.fit_transform(list(voters_and_non_voters["educ"]))
income_cat = le.fit_transform(list(voters_and_non_voters["income_cat"]))
voter_categories = ['always', 'sporadic', 'rarely/never']
voter_category = [voter_categories.index(category) if category in voter_categories else -1 for category in voters_and_non_voters["voter_category"]]
y = np.array(voter_category)

x = list(zip(RespId, gender, race, educ, income_cat))
y = list(zip(voter_category))
num_folds = 5
seed = 7
scoring = 'accuracy'
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.20, random_state=seed)
np.shape(x_train), np.shape(x_test)


dataset = pd.read_csv('nonvoters_dataset.csv')
dataset = pd.get_dummies(dataset)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
models = []
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))

# set number of folds for cross-validation
num_folds = 5

# set random seed for reproducibility
seed = 7

# evaluate each model in turn
results = []
names = []
print("Performance on Training set")
for name, model in models:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, x, y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = f"{name}: {cv_results.mean()} ({cv_results.std()})"
    print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


dt = DecisionTreeClassifier()
nb = GaussianNB()
gb = GradientBoostingClassifier()
rf = RandomForestClassifier()
best_model = rf
y_train = np.array(y_train)
best_model.fit(x_train, y_train.ravel())
y_pred = best_model.predict(x_test)
print("Best Model Accuracy Score on Test Set:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)
y_onehot_test.shape

class_id = 2
class_of_interest = "Always"
RocCurveDisplay.from_predictions(y_onehot_test[:, class_id], y_pred,
                                 name=f"{class_of_interest} vs the rest", color="darkorange")
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\n Always vs (rarely/never & sporadic)")
plt.legend()
plt.show()

correct = 0
total = len(y_test)

for i in range(len(y_test)):
    predicted_class = y_pred[i]
    actual_class = y_test[i][0]
    print("Predicted:", predicted_class, "Actual:", actual_class, "Data:", x_test[i])











