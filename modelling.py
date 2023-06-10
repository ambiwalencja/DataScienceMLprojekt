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

# PCA
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

# SMOTE balancing
from imblearn.over_sampling import SMOTE

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


def compare_results(method, score):
    global results  # trzeba to dodać, żeby interpretter wiedział, że results to jest zmienna globalna i żeby nie
    # próbował sie od niej odwołac jako do lokalneji żeby nie wyrzucał błędu
    # "UnboundLocalError: local variable 'results' referenced before assignment

    temp_results = pd.DataFrame({'Method': method,
                                 'ROC AUC score': round(score, 3)},
                                index=[len(results.index) + 1])
    results = pd.concat([results, temp_results])
    return results


def standarize_dataset(array_like_object):
    scaled_array = scaler.fit_transform(array_like_object)
    return scaled_array


def perform_pca(array_like_object):
    global pca_final
    scaled_array = standarize_dataset(array_like_object)
    pca = PCA()
    pca.fit(scaled_array)
    var_cumu = np.cumsum(pca.explained_variance_ratio_)
    for i, sum in enumerate(var_cumu):
        if sum >= 0.95:
            pca_final = IncrementalPCA(n_components=i)
            break
    return pca_final.fit_transform(scaled_array)


def perform_smote(x, y):
    sm = SMOTE(random_state=42)
    return sm.fit_resample(x, y)


class Model:
    def __init__(self, x, y, model_class):
        self.auroc_mean = 0
        self.list_auroc_stratified = []
        self.df_x = x
        self.df_y = y
        self.model = model_class

    def print_results(self):
        print('List of AUROC scores:', self.list_auroc_stratified)
        self.auroc_mean = np.mean(self.list_auroc_stratified)
        print('\nMean AUROC:', round(self.auroc_mean * 100, 3), '%')

    def train_model_kfold(self):  # no smote
        array_x = self.df_x.to_numpy()
        array_y = self.df_y.to_numpy()
        for train_index, test_index in skf.split(self.df_x, self.df_y):
            x_train_fold, x_test_fold = array_x[train_index], array_x[test_index]
            y_train_fold, y_test_fold = array_y[train_index], array_y[test_index]

            scaled_x_array_train_fold = standarize_dataset(x_train_fold)
            scaled_x_array_test_fold = standarize_dataset(x_test_fold)

            self.model.fit(scaled_x_array_train_fold, np.ravel(y_train_fold))
            self.list_auroc_stratified.append(
                roc_auc_score(y_test_fold, self.model.predict_proba(scaled_x_array_test_fold)[:, 1]))
        self.print_results()

    def train_model_pca_kfold(self):  # no smote
        array_x = self.df_x.to_numpy()
        array_y = self.df_y.to_numpy()
        for train_index, test_index in skf.split(self.df_x, self.df_y):
            x_train_fold, x_test_fold = array_x[train_index], array_x[test_index]
            y_train_fold, y_test_fold = array_y[train_index], array_y[test_index]

            scaled_x_array_train_fold = perform_pca(x_train_fold)
            scaled_x_array_test_fold = perform_pca(x_test_fold)

            self.model.fit(scaled_x_array_train_fold, np.ravel(y_train_fold))
            self.list_auroc_stratified.append(
                roc_auc_score(y_test_fold, self.model.predict_proba(scaled_x_array_test_fold)[:, 1]))
        self.print_results()

    def train_model(self):
        train_x, test_x, train_y, test_y = train_test_split(self.df_x, self.df_y, test_size=0.3, random_state=0,
                                                            stratify=self.df_y['target'])
        scaled_train_x_array = standarize_dataset(train_x)
        train_x, train_y = perform_smote(scaled_train_x_array, train_y)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        # ssp = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        self.list_auroc_stratified = cross_val_score(self.model, train_x, np.ravel(train_y),
                                                     # DataConversionWarning: A column-vector y was passed when a 1d
                                                     # array was expected. Please change the shape of y to (n_samples, )
                                                     # , for example using ravel().
                                                     #   y = column_or_1d(y, warn=True)
                                                     # dlatego zamiast train_y jest np.ravel(train_y)
                                                     scoring='roc_auc', cv=skf)  # albo: cv=ssp
        self.print_results()

    def train_model_pca(self):
        train_x, test_x, train_y, test_y = train_test_split(self.df_x, self.df_y, test_size=0.3, random_state=0,
                                                            stratify=self.df_y['target'])
        scaled_train_x_array = perform_pca(train_x)
        train_x, train_y = perform_smote(scaled_train_x_array, train_y)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        self.list_auroc_stratified = cross_val_score(self.model, train_x, np.ravel(train_y),
                                                     scoring='roc_auc', cv=skf)
        self.print_results()

    def train_model_grid_search(self, grid):
        train_x, test_x, train_y, test_y = train_test_split(self.df_x, self.df_y, random_state=0, stratify=self.df_y)
        scaled_train_x_array = standarize_dataset(train_x)
        train_x, train_y = perform_smote(scaled_train_x_array, train_y)
        skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=self.model, param_grid=grid, n_jobs=-1, cv=skf,
                                   scoring='roc_auc', error_score=0)
        grid_result = grid_search.fit(train_x, np.ravel(train_y))

        print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

# __________________________________________________________________________________________________________


# for tests
df = pd.read_csv("Loan_data_new_variables.csv")
myModel = Model(df.drop('target', axis=1), df[['target']], LogisticRegression(max_iter=1000))
# myModel.train_model_pca()
myModel.train_model_grid_search({'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                                 'C': [100, 10, 1.0, 0.1, 0.01]})
# print(compare_results('Logistic Regression', myModel.auroc_mean))
