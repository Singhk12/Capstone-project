import os
import numpy
import pickle
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import shuffle
from sklearn import datasets
import matplotlib.pyplot as pyplot
from matplotlib import style
from sklearn import svm
from sklearn import linear_model, preprocessing
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import set_option
from pandas_profiling import ProfileReport
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
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
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.exceptions import DataDimensionalityWarning
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

# Data file import
df = pd.read_csv("nonvoters_dataset.csv")

fig = plt.figure(figsize=(16, 16))
ax = fig.gca()
df.hist(ax=ax, bins=30)
plt.show()

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(16, 16))
axs = axs.flatten()

for i, col in enumerate(['educ', 'race', 'gender', 'income_cat', 'voter_category']):
    sns.countplot(x=col, data=df, ax=axs[i])
    axs[i].set_title(col)

plt.tight_layout()
plt.show()

continous_features = ['RespId']
categorical_features = ['educ', 'race', 'gender', 'income_cat', 'voter_category']


def outliers(df_out, drop=False):
    for each_feature in df_out.columns:
        feature_data = df_out[each_feature]
        if each_feature in categorical_features:
            print('For the feature {}, No outliers detected'.format(each_feature))
        else:
            Q1 = np.percentile(feature_data, 25.)
            Q3 = np.percentile(feature_data, 75.)
            IQR = Q3 - Q1
            outlier_step = IQR * 1.5
            outliers = feature_data[
                ~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))].index.tolist()
            if len(outliers) > 0:
                print('For the feature {}, Number of outliers is {}'.format(each_feature, len(outliers)))
                if drop:
                    df.drop(outliers, inplace=True, errors='ignore')
                    print('Outliers from {} feature removed'.format(each_feature))
            else:
                print('For the feature {}, No outliers detected'.format(each_feature))


outliers(df[continous_features + categorical_features])

outliers(df[continous_features], drop=True)

df.plot(kind='box', subplots=True, layout=(2, 7), sharex=False, sharey=False, figsize=(20, 10), color='deeppink');


df = df[df.voter_category != 'e']
# specify the tick labels using `name`
name = ["always", "rarely/never", "sporadic"]
# plot the data
fig, ax = plt.subplots(figsize=(7, 6))
ax = df.voter_category.value_counts().plot(kind='bar')
ax.set_title("Voters and Non Voters", fontsize=12, weight='bold')
ax.set_xticklabels(name, rotation=0)
totals = []
for i in ax.patches:
    totals.append(i.get_height())
total = sum(totals)
for i in ax.patches:
    ax.text(i.get_x() + 0.07, i.get_height() - 160, \
            str(round((i.get_height() / total) * 100, 2)) + '%', fontsize=14, color='white', weight='bold')
plt.tight_layout()
plt.show()

df = pd.get_dummies(df, columns=['gender', 'race', 'educ', 'income_cat', 'voter_category'])
# Compute the correlation matrix
corr = df.corr()
# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True, cmap='Blues')
plt.show()












