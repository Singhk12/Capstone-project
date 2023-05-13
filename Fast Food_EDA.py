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

df = pd.read_csv("fastfood.csv")

fig = plt.figure(figsize=(16, 16))
ax = fig.gca()
df.hist(ax=ax, bins=30)
plt.show()

# Create subplots for each category (objects)
fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(16, 16))
axs = axs.flatten()

for i, col in enumerate(['restaurant','salad']):
    if not df[col].empty:
        sns.countplot(x=col, data=df, ax=axs[i])
        axs[i].set_title(col)

plt.tight_layout()
plt.show()

continous_features = ['calories', 'cal_fat', 'total_fat', 'sat_fat','trans_fat', 'cholesterol', 'sodium', 'total_carb', 'sugar']
categorical_features = ['restaurant', 'item', 'salad']

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
            outliers = feature_data[~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))].index.tolist()
            if len(outliers) > 0:
                print('For the feature {}, Number of outliers is {}'.format(each_feature, len(outliers)))
                if drop:
                    df.drop(outliers, inplace=True, errors='ignore')
                    print('Outliers from {} feature removed'.format(each_feature))
            else:
                print('For the feature {}, No outliers detected'.format(each_feature))

outliers(df[continous_features + categorical_features])

outliers(df[continous_features], drop = True)

df.plot(kind='box', subplots=True,
layout=(2,7),sharex=False,sharey=False, figsize=(20, 10), color='deeppink');


fig, ax = plt.subplots(figsize=(5, 4))
name = ["Arbys", "Burger King", "Chick-Fil-A", "Dairy Queen", "Mcdonalds", "Sonic", "Taco Bell", "Subway"]
ax = df.restaurant.value_counts().plot(kind='bar')
ax.set_title("Fast Food Nutrition", fontsize=13, weight='bold')
ax.set_xticklabels(name, rotation=45)

# To calculate the percentage
totals = []
for i in ax.patches:
    totals.append(i.get_height())
total = sum(totals)
for i in ax.patches:
    ax.text(i.get_x() + 12.0, i.get_height() - 60, \
            str(round((i.get_height() / total) * 200, 20)) + '%', fontsize=14,
            color='white', weight='bold')

plt.tight_layout()

sns.set(style="white")
plt.rcParams['figure.figsize'] = (15, 10)
sns.heatmap(df.corr(), annot = True, linewidths=.5, cmap="Blues")
plt.title('Correlation Between Variables', fontsize = 30)
plt.show()