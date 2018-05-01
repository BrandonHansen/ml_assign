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

print("collecting data...")

full_feature_cols = [i for i in range(0, 48)]
cont_feature_cols = [3,14,24,39]
disc_feature_cols = [i for i in range(0, 48) if i not in cont_feature_cols]
class_labels = 48

full_features = pd.read_csv(file_name, usecols=full_feature_cols)
cont_features = pd.read_csv(file_name, usecols=cont_feature_cols)
disc_features = pd.read_csv(file_name, usecols=disc_feature_cols)
classes = pd.read_csv(file_name, usecols=[class_labels])

selected_features = disc_features

features_data = selected_features.as_matrix()
class_data = np.ravel(classes.as_matrix())

skf = StratifiedKFold(n_splits=10)

print("beginning training/testing with bayes...")

scores = 0

for train_index, test_index in skf.split(features_data, class_data):
	feat_train, feat_test = features_data[train_index], features_data[test_index]
	class_train, class_test = class_data[train_index], class_data[test_index]
	
	clf = GaussianNB()
	clf.fit(feat_train, class_train)
	predicted = clf.predict(feat_test)
	
	score = accuracy_score(class_test, predicted)
	scores = scores + score

print("average: ", scores/10)

print("beginning training/testing with knn...")

scores = 0

for train_index, test_index in skf.split(features_data, class_data):
	feat_train, feat_test = features_data[train_index], features_data[test_index]
	class_train, class_test = class_data[train_index], class_data[test_index]
	
	clf = KNeighborsClassifier(n_neighbors=3)
	clf.fit(feat_train, class_train)
	predicted = clf.predict(feat_test)
	
	score = accuracy_score(class_test, predicted)
	scores = scores + score

print("average: ", scores/10)

print("beginning training/testing with rf...")

scores = 0

for train_index, test_index in skf.split(features_data, class_data):
	feat_train, feat_test = features_data[train_index], features_data[test_index]
	class_train, class_test = class_data[train_index], class_data[test_index]
	
	clf = RandomForestClassifier(max_depth=2, random_state=0)
	clf.fit(feat_train, class_train)
	predicted = clf.predict(feat_test)
	
	score = accuracy_score(class_test, predicted)
	scores = scores + score

print("average: ", scores/10)
	
print("beginning training/testing with svm...")

scores = 0

for train_index, test_index in skf.split(features_data, class_data):
	feat_train, feat_test = features_data[train_index], features_data[test_index]
	class_train, class_test = class_data[train_index], class_data[test_index]
	
	clf = svm.SVC()
	clf.fit(feat_train, class_train)
	predicted = clf.predict(feat_test)
	
	score = accuracy_score(class_test, predicted)
	scores = scores + score

print("average: ", scores/10)

print("beginning training/testing with mlp...")

scores = 0

for train_index, test_index in skf.split(features_data, class_data):
	feat_train, feat_test = features_data[train_index], features_data[test_index]
	class_train, class_test = class_data[train_index], class_data[test_index]
	
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(feat_train, class_train)
	predicted = clf.predict(feat_test)
	
	score = accuracy_score(class_test, predicted)
	scores = scores + score

print("average: ", scores/10)