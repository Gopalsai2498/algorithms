# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 23:30:37 2020

@author: anil.ms
"""
#detailed explanation
#https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/


# logistic regression for multi-class classification using built-in one-vs-rest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)

# define model
model = LogisticRegression(multi_class='ovr')

# fit model
model.fit(X, y)

# make predictions
yhat = model.predict(X)

feature_importance = pd.DataFrame(sorted(zip(model.feature_importances_, trainset.columns)), columns=['Value','Feature'])

plt.figure(figsize=(10, 6))
sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False).head(20))
plt.title('Variable Importance plot ')
plt.tight_layout()

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

fig, axes = plt.subplots(figsize=(12, 6))
sns.scatterplot(Y_predicted, Y_test)
plt.xlabel('Predicted Sales')
plt.ylabel('Sales')
plt.title('Actual vs predicted sales', fontsize=16)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
rmse = np.sqrt(mean_squared_error(yhat, y))
mae = mean_absolute_error(yhat, y)
r2 = r2_score(yhat, y)


print("train rmse: %f" % rmse)
print("train mae: %f" %mae)
print("train r2: %f" %r2)




# logistic regression for multi-class classification using a one-vs-rest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)

# define model
model = LogisticRegression()

# define the ovr strategy
ovr = OneVsRestClassifier(model)  ################

# fit model
ovr.fit(X, y)

# make predictions
yhat = ovr.predict(X)



#One-Vs-One for Multi-Class Classification
#One-vs-One (OvO for short) is another heuristic method for using binary classification algorithms for multi-class classification.

# SVM for multi-class classification using built-in one-vs-one
from sklearn.datasets import make_classification
from sklearn.svm import SVC
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
model = SVC(decision_function_shape='ovo')
# fit model
model.fit(X, y)
# make predictions
yhat = model.predict(X)


# SVM for multi-class classification using one-vs-one
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)
# define model
model = SVC()

# define ovo strategy
ovo = OneVsOneClassifier(model)
# fit model
ovo.fit(X, y)
# make predictions
yhat = ovo.predict(X)








