import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Bibliotheque pour le Machine Learning

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Data importation

data =  pd.read_excel("C:\\Users\\Ulysse K\\cardio prediction\\data\\raw\\cardio.xlsx")
data.head()

df = data.copy('deep')

# features extraction
features = df.drop('cardio', axis=1)
target = df['cardio']


