import scipy
import numpy as np
import matplotlib

# pyplot
from matplotlib import pyplot as plt

# pandas
import pandas as pd
from pandas.plotting import scatter_matrix

# scikit-learn module
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
"""
str: The link to the dataset.
"""
names = [
    "id",
    "clump_thickness",
    "uniform_cell_size",
    "uniform_cell_shape",
    "marginal_adhesion",
    "single_epithelial_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
    "class"
]
"""
list[str]: The column heading I want to get.
"""
df = pd.read_csv(
    url,
    names = names
)
"""
pandas.core.frame.DataFrame: The DataFrame with my data.
"""
df


# Preprocess the data
find_what = "?"
"""
str: The old entry for missing data.
"""
replace_with = -99999
"""
int: The new entry for missing data.
"""
df.replace(
    find_what,
    replace_with,
    inplace=True
)
column_to_drop = ['id']
df.drop(
    column_to_drop,
    1,
    inplace=True
)
df


# Let explore the dataset and do a few visualizations
row_to_print = 3
"""
int: The row to print in this example.
"""
print(df.shape)
df.loc[row_to_print]


# Describe the dataset
df.describe()


# Plot histograms for each variable.
my_figsize = (10, 10)
"""
tuple[int, int]: The size of the histograms I want to print.
"""
df.hist(figsize = my_figsize)
plt.show()


# Create a scatter plot matrix
my_figsize = (18, 18)
scatter_matrix(
    df,
    figsize=my_figsize
)
plt.show()



