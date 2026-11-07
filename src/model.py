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
features = df.drop(['cardio', 'id'], axis=1)
target = df['cardio']

print(features.head())
# train test split

X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.2, random_state=42)


# Dictonnary hyperparams

model_params = {
    'Logistic_Regression' : {
        'model' : LogisticRegression(),
        'params' : {}
    },

    'Decision_Tree_Classifier' : {
        'model' : DecisionTreeClassifier(),
        'params' : {
            'max_depth' : [2,4,6,8,10],
            'random_state': [0, 42],
            'min_samples_split': [1, 5, 10, 20]


        }
    },

    'Random_Forest_Classifier' : {
        'model' : RandomForestClassifier(),
        'params' : {
            'n_estimators' : [10, 20, 30, 50, 80]
        }
    }
}

# Grid search cv

score = []
for model_name, m in model_params.items():
    clf = GridSearchCV(m['model'], m['params'], cv=5, scoring='neg_mean_squared_error')
    clf.fit(X_train, y_train)

    score.append({
        'Model': model_name,
        'Params' : clf.best_params_,
        'MSE(-ve)': clf.best_score_
    })

pd.DataFrame(score)
print(score)

# best model
s = pd.DataFrame(score)
sort = s.sort_values(by= 'MSE(-ve)', ascending=False)
print(sort)

# 1. Random forest model
print("=== Random Forest Model ==\n\n")
rfr = RandomForestClassifier(n_estimators=80)
rfr.fit(X_train, y_train)

print(rfr.score(X_test, y_test))
y_pred_rfr = rfr.predict(X_test)

print(y_pred_rfr)

# Metrics
print("Mean Squared Error : ", mean_squared_error(y_test, y_pred_rfr))
print("Mean Absolute Error : ", mean_absolute_error(y_test, y_pred_rfr))

# 2. Decision Tree Model
print("\n\n== Decision Tree Model ==\n\n")

dtr = DecisionTreeClassifier(max_depth=6, min_samples_split=5, random_state=42)
dtr.fit(X_train, y_train)

print(dtr.score(X_test, y_test))
y_pred_dtr = dtr.predict(X_test)
print(y_pred_dtr)

# Metrics
print("Mean Squared Error : ", mean_squared_error(y_test, y_pred_dtr))
print("Mean Absolute Error : ", mean_absolute_error(y_test, y_pred_dtr))

# Logistic classifier Model
print("\n\n== Logistic Regression Model ==\n\n")

lc = LogisticRegression()
lc.fit(X_train, y_train)

print(lc.score(X_test, y_test))
y_pred_lc = lc.predict(X_test)
print(y_pred_lc)

# Metrics
print("Mean Squared Error : ", mean_squared_error(y_test, y_pred_lc))
print("Mean Absolute Error : ", mean_absolute_error(y_test, y_pred_lc))

# Diagramme de barre importances des caracteristiques Random Forest
feature_importance = rfr.feature_importances_
features_names = list(X_train.columns)

sorted_indices = np.argsort(feature_importance)[::-1]
sorted_feature_importance = [feature_importance[i] for i in sorted_indices]
sorted_feature_names = [features_names[i] for i in sorted_indices]


plt.figure(figsize=(12, 8))
plt.barh(sorted_feature_names[:10], sorted_feature_importance[:10])
plt.xlabel("Features Importance")
plt.title('Top 10 Feature importance in predicting Cardio')
plt.gca().invert_yaxis()

plt.show()

# Overfitting and underfitting analysis

print("\n=== Analyse du sur-apprentissage/sous apprentissage ===\n")

# Evaluation de la profondeur de l'arbre sur la performance
max_depths = range(1,21)
train_scores_dt = []
test_scores_dt = []

for depth in max_depths:
    dt_temp = DecisionTreeClassifier(max_depth=depth, random_state=2)
    dt_temp.fit(X_train, y_train)
    train_scores_dt.append(dt_temp.score(X_train, y_train))
    test_scores_dt.append(dt_temp.score(X_test, y_test))

# Visualisation la performance en fonctoion de la profondeur
plt.figure(figsize=(10,6))
plt.plot(max_depths, train_scores_dt, 'b-', label='Score entrainement')
plt.plot(max_depths, test_scores_dt, 'r-', label='Score test')
plt.xlabel('Profondeur maximale de l\'arbre')
plt.ylabel('Precision')
plt.title('Analyse du sur-apprentissage/sous-apprentissage')
plt.legend()
plt.grid()
plt.show()
