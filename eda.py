import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pandas.api.types import is_numeric_dtype, \
    is_object_dtype  # do robienia listy zmiennych numerycznych/kategorycznych
import time  # przydaje się


# funkcja do wyświetlenia rozkładu
def print_distribution_in_percents(df, variable):
    values_distribution = df[variable].value_counts().sort_index()
    print(round(100 * (values_distribution / len(df.index)), 2))


# funkcja wyznaczająca outliers
def outliers(df, clmn):
    print(f'Number of outliers in {clmn}: ',
          df[((df[clmn] - df[clmn].mean()) /
              df[clmn].std()).abs() > 3][clmn].count())


# wykresy
def plots(df, clmn):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    sns.boxplot(x=clmn, data=df, ax=ax1)
    sns.boxplot(data=df, x='target', y=clmn, ax=ax2)
    sns.histplot(x=clmn, data=df, ax=ax3)
    sns.histplot(data=df, x=clmn, hue="target", ax=ax4)
    fig.suptitle(clmn, fontsize=16)


# porównanie średnich zmiennej w podziale na podgrupy target
def compare_means(df, clmn):
    print(f'Średnia {clmn} w podgrupach targetu:')
    print(df.groupby('target')[clmn].mean())


def create_numeric_column_list(df):
    numeric_column_list = []
    for clmn in df.columns.tolist():
        if is_numeric_dtype(df[clmn]):
            numeric_column_list.append(clmn)
    return numeric_column_list


# information source regarding T-test and how the variances are treated as equal:
# "Applied Linear Statistical Models" by Michael H. Kutner, Christopher J. Nachtsheim, John Neter, and William Li.
# In the book, they suggest that if the ratio of the larger variance to the smaller variance is no more than 4:1
# or 3:1, then the variances can be considered approximately equal for the purposes of the t-test.
#
# Kutner, Nachtsheim, Neter, and Li. (2005). Applied Linear Statistical Models (5th Edition).
# McGraw-Hill Higher Education.
# https://www.statology.org/determine-equal-or-unequal-variance/

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html#scipy.stats.ttest_ind
def perform_ttest(df, binary_var, checked_var):
    variance_0 = df.loc[df[binary_var] == 0][checked_var].var()
    variance_1 = df.loc[df[binary_var] == 1][checked_var].var()
    if (variance_0 / variance_1 > 3) | (variance_0 / variance_1 < 1 / 3):
        eqv = False
    else:
        eqv = True
    stat, p = stats.ttest_ind(df.loc[df[binary_var] == 0][checked_var],
                              df.loc[df[binary_var] == 1][checked_var], equal_var=eqv)
    print(f"T-test result: statistic = {stat}, p value =  {p}")


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html#scipy.stats.mannwhitneyu
def perform_mannwhitneyutest(df, binary_var, checked_var):
    stat, p = stats.mannwhitneyu(df.loc[df[binary_var] == 0][checked_var], df.loc[df[binary_var] == 1][checked_var])
    print(f"Mann-Whitney U test result: result: statistic = {stat}, p value = {p}")

# -----------------------------------------------------------------------------------------------
# # for tests:
# my_df = pd.read_csv("Loan_data_rob.csv")
# my_df = pd.read_csv("Loan_data_after_eda_part.csv")

# perform_ttest(my_df, 'target', 'loan_amnt')

# column_list = create_numeric_column_list(my_df)
# for column in column_list:
#     print(column.upper())
#     print(my_df[column].describe().T)
#     outliers(my_df, column)
#     # plots(my_df, column)
#     compare_means(my_df, column)
#     print('\n\n')
