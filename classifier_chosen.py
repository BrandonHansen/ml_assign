from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

file_name = "classification_train.txt"

print("collecting data for training...")

full_feature_cols = [i for i in range(0, 48)]
cont_feature_cols = [3,14,24,39]
disc_feature_cols = [i for i in range(0, 48) if i not in cont_feature_cols]
class_labels = 48

full_features = pd.read_csv(file_name, usecols=full_feature_cols)
cont_features = pd.read_csv(file_name, usecols=cont_feature_cols)
disc_features = pd.read_csv(file_name, usecols=disc_feature_cols)
classes = pd.read_csv(file_name, usecols=[class_labels])

selected_features = cont_features

features_data = selected_features.as_matrix()
class_data = np.ravel(classes.as_matrix())

clf = svm.SVC()
clf.fit(features_data, class_data)


predicted = clf.predict(features_data)
score = accuracy_score(class_data, predicted)
print("resubstitution score:", score)

file_name = "classification_test.txt"

print("collecting data for testing...")

full_feature_cols = [i for i in range(0, 48)]
cont_feature_cols = [3,14,24,39]
disc_feature_cols = [i for i in range(0, 48) if i not in cont_feature_cols]

full_features = pd.read_csv(file_name, usecols=full_feature_cols)
cont_features = pd.read_csv(file_name, usecols=cont_feature_cols)
disc_features = pd.read_csv(file_name, usecols=disc_feature_cols)
classes = pd.read_csv(file_name, usecols=[class_labels])

selected_features = cont_features

features_data = selected_features.as_matrix()
class_data = np.ravel(classes.as_matrix())

predicted = clf.predict(features_data)
score = accuracy_score(class_data, predicted)

predicted = clf.predict(features_data)
score = accuracy_score(class_data, predicted)
print("nonsense score:", score)

predicted.astype(int)

np.savetxt("classification_results.csv", predicted, delimiter=",")