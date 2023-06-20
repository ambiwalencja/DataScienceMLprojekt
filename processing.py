# remember to start with pip install of packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pandas.api.types import is_numeric_dtype, \
    is_object_dtype  # do robienia listy zmiennych numerycznych/kategorycznych
import time  # przydaje się


def drop_constant_columns(df):
    """drops any columns from df, that have only one value"""
    return df.loc[:, (df != df.iloc[0]).any()]


def print_sorted_values(df, column_list):
    """prints values of variables given in column_list, ordered ascending"""
    for column in column_list:
        print(column, '\n', (df[column].value_counts(normalize=True) * 100).sort_values(), '\n\n')


def create_10_categories(df, column, new_column_name):
    """creates a ten-categories variable named new_column_name, from a variable column,
    based on 10 quantiles of variable column"""
    quantiles = [0]
    for quantile in range(1, 10):
        quantiles.append(df[column].quantile(q=quantile / 10))

    def map_values(value):
        if value < quantiles[1]:
            return 1
        elif value < quantiles[2]:
            return 2
        elif value < quantiles[3]:
            return 3
        elif value < quantiles[4]:
            return 4
        elif value < quantiles[5]:
            return 5
        elif value < quantiles[6]:
            return 6
        elif value < quantiles[7]:
            return 7
        elif value < quantiles[8]:
            return 8
        elif value < quantiles[9]:
            return 9
        else:
            return 10
    df[new_column_name] = df[column].apply(lambda value: map_values(value))


# funkcja do wyświetlenia rozkładu
def print_distribution_in_percents(df, variable):
    """prints the distribution of variable, sorted by frequency, in percents"""
    values_distribution = df[variable].value_counts().sort_index()
    print(round(100 * (values_distribution / len(df.index)), 2))


# funkcja wyznaczająca outliers
def outliers(df, clmn, thresh):
    """Returns the number of outliers for the given variable list, outliers understood as observations that are
    beyond the certain threshold value after performing a standardization on the variable. It is common to consider
     an observation an outlier if after standardization its absolute value is greater than 3."""
    print(f'Number of outliers in {clmn}: ',
          df[((df[clmn] - df[clmn].mean()) /
              df[clmn].std()).abs() > thresh][clmn].count())


# wykresy
def plots(df, clmn, group_var):
    """Creates 4 plots:
     - a boxplot of a variable clmn
     - a boxplot of a variable clmn grouped by group_var
     - a histogram of variable clmn
     - a histogram of a variable clmn grouped by group_var"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    sns.boxplot(x=clmn, data=df, ax=ax1)
    sns.boxplot(data=df, x=group_var, y=clmn, ax=ax2)
    sns.histplot(x=clmn, data=df, ax=ax3)
    sns.histplot(data=df, x=clmn, hue=group_var, ax=ax4)
    fig.suptitle(clmn, fontsize=16)


# porównanie średnich zmiennej w podziale na podgrupy target
def compare_means(df, clmn, group_var):
    """prints means of clmn grouped by group_var variable"""
    print(f'Średnia {clmn} w podgrupach {group_var}:')
    print(df.groupby(group_var)[clmn].mean())


def create_numeric_column_list(df):
    """creates a list of all variables from dataframe df, that are numeric"""
    numeric_column_list = []
    for clmn in df.columns.tolist():
        if is_numeric_dtype(df[clmn]):
            numeric_column_list.append(clmn)
    return numeric_column_list


def display_analysis(df, column_list, group_var):
    """joints above functions and creates full analysis of each variable from column_list"""
    for column in column_list:
        print(column.upper())
        print(df[column].describe().T)
        outliers(df, column, 3)
        plots(df, column, group_var)
        compare_means(df, column, group_var)
        print('\n\n')

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
    """performs Student's t-test, comparing means of variable checked_var in two groups of binary_var. Given that this
    test assumes that the variances of two groups are equal, I calculate the variances (variance_0 and variance_1)
    and compare them. According to "Applied Linear Statistical Models" by Michael H. Kutner, Christopher J. Nachtsheim,
     John Neter, and William Li "if the ratio of the larger variance to the smaller variance is no more than 4:1
    or 3:1, then the variances can be considered approximately equal for the purposes of the t-test" so I check if the
    variances meet the condition, if so, I perform standard t-test, if not, I set the "equal_vavr" parameter to false
    and thereby perform Welch’s t-test instead."""
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
    """performs Mann-Whitney U-test, comparing medians of variable checked_var in two groups of binary_var."""
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
