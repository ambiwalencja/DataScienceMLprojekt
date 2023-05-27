import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# statystyki, metryki
from statistics import mean, stdev
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# podział zbioru
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

# standaryzacja
from sklearn.preprocessing import StandardScaler

# modele
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# optymalizacja
from sklearn.model_selection import GridSearchCV

# -------------------------------------------------------------------------------------------------------------
# global variables
scaler = StandardScaler()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
results = pd.DataFrame()


def compare_results(method, auroc_mean):
    temp_results = pd.DataFrame({'Method': [method], 'ROC AUC score': round(auroc_mean, 3)}, index=['2'])
    return pd.concat([results, temp_results])


class Model:
    def __init__(self, x, y):
        self.auroc_mean = 0
        self.list_auroc_stratified = []
        self.df_x = x
        self.df_y = y

    def standarize_dataset(self, array_like_object):
        scaled_array = scaler.fit_transform(array_like_object)
        return scaled_array

    def print_results(self):
        print('List of AUROC scores:', self.list_auroc_stratified)
        self.auroc_mean = np.mean(self.list_auroc_stratified)
        print('\nMean AUROC:', round(self.auroc_mean * 100, 3), '%')

    def train_model(self, model_class):
        model = model_class
        self.list_auroc_stratified = []
        array_x = self.df_x.to_numpy()
        array_y = self.df_y.to_numpy()
        for train_index, test_index in skf.split(self.df_x, self.df_y):
            x_train_fold, x_test_fold = array_x[train_index], array_x[test_index]
            y_train_fold, y_test_fold = array_y[train_index], array_y[test_index]

            scaled_x_array_train_fold = self.standarize_dataset(x_train_fold)
            scaled_x_array_test_fold = self.standarize_dataset(x_test_fold)

            model.fit(scaled_x_array_train_fold, np.ravel(y_train_fold))
            self.list_auroc_stratified.append(
                roc_auc_score(y_test_fold, model.predict_proba(scaled_x_array_test_fold)[:, 1]))
        self.print_results()

# __________________________________________________________________________________________________________


# def standarize_dataset(array_like_object):
#     scaled_array = scaler.fit_transform(array_like_object)
#     return scaled_array
#
#
# def print_results(list_of_results):
#     print('List of AUROC scores:', list_of_results)
#     auroc_mean = np.mean(list_of_results)
#     print('\nMean AUROC:', round((auroc_mean) * 100, 3), '%')
#
#
# def train_model(model_class, df_x, df_y):
#     model = model_class
#     list_auroc_stratified = []
#     array_x = df_x.to_numpy()
#     array_y = df_y.to_numpy()
#     for train_index, test_index in skf.split(df_x, df_y):
#         x_train_fold, x_test_fold = array_x[train_index], array_x[test_index]
#         y_train_fold, y_test_fold = array_y[train_index], array_y[test_index]
#
#         scaled_x_array_train_fold = standarize_dataset(x_train_fold)
#         scaled_x_array_test_fold = standarize_dataset(x_test_fold)
#
#         model.fit(scaled_x_array_train_fold, np.ravel(y_train_fold))
#         list_auroc_stratified.append(
#             roc_auc_score(y_test_fold, model.predict_proba(scaled_x_array_test_fold)[:, 1]))
#     print_results(list_auroc_stratified)

# ------------------------------------------------------------------------------------------------

# for tests
df = pd.read_csv("Loan_data_new_variables.csv")
myModel = Model(df.drop('target', axis=1), df[['target']])
myModel.train_model(LogisticRegression(max_iter=500))
compare_results('Logistic Regression', myModel.auroc_mean)