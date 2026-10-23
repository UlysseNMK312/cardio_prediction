# Data preprocessing 

# IMPORTATION DES BIBLIOTHEQUES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

 # IMPORTATION DU JEU DE DONNEES
data =  pd.read_excel("C:\\Users\\Ulysse K\\cardio prediction\\data\\raw\\cardio.xlsx")
data.head()

# BASIC INFORMATION

# data information
data.info()

# dimension du dataset
data.shape

# columns names ( variables)
data.columns.unique()

# data types
data.dtypes

# statistics descriptive
data.describe()

# DATA CLEANING

# fixe issues in data
# check for NaN
data.isnull().sum()


