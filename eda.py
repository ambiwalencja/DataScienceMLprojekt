import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from pandas.api.types import is_numeric_dtype, is_object_dtype # do robienia listy zmiennych numerycznych/kategorycznych

import time # przydaje się


# funkcja wyznaczająca outliersy
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

# -----------------------------------------------------------------------------------------------


my_df = pd.read_csv("Loan_data_rob.csv")
column_list = create_numeric_column_list(my_df)
for column in column_list:
    print(column.upper())
    print(my_df[column].describe().T)
    outliers(my_df, column)
    # plots(my_df, column)
    compare_means(my_df, column)
    print('\n\n')
