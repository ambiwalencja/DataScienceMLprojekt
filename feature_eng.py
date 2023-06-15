import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def find_outliers_method_normal_distribution(df, col_list, threshold):
    """Returns the number of outliers for the given variable list, outliers understood as observations that are
    beyond the certain threshold value after performing a standardization on the variable. It is common to consider
     an observation an outlier if after standardization its absolute value is greater than 3."""
    for column in col_list:
        print(f'Number of outliers in {column}: ',
              df.loc[(((df[column] - df[column].mean()) / df[column].std()).abs() > threshold)][column].count())


def replace_outliers(df, column_list, column_groupby_list):
    """Replaces the outliers in columns from the column_list with mean values by categories created with
    column_groupby_list"""
    def replace(group):
        mean, std = group.mean(), group.std()
        outliers = (group - mean).abs() > 3*std
        group[outliers] = mean
        return group
    df[column_list] = df.groupby(column_groupby_list)[column_list].transform(replace)


def log_plus_1_transform(df, column_list):
    def logarithm(df_, column_):
        name = column_+"_log+1"
        df_[name] = (df_[column_]+1).transform(np.log)

    for column in column_list:
        logarithm(df, column)


def create_binary_variables_with_median(df, column_list):
    """Create binary variables from the list of given variables, with median as a threshold"""
    for column in column_list:
        name = column+'_bin'
        df[name] = (df[column] >= df[column].median()).astype(int)


# print(find_outliers_method_normal_distribution.__doc__)