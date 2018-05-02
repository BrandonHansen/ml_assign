from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

file_name = "regression_train.data"

print("collecting data...")

full_feature_cols = [i for i in range(0, 21)]
class_labels = 21

full_features = pd.read_csv(file_name, usecols=full_feature_cols)
classes = pd.read_csv(file_name, usecols=[class_labels])

selected_features = full_features

features_data = selected_features.as_matrix()
class_data = np.ravel(classes.as_matrix())

skf = StratifiedKFold(n_splits=10)

print("beginning training/testing with knn...")

scores = 0

for train_index, test_index in skf.split(features_data, class_data):
	feat_train, feat_test = features_data[train_index], features_data[test_index]
	class_train, class_test = class_data[train_index], class_data[test_index]
	
	clf = KNeighborsRegressor(n_neighbors=3)
	clf.fit(feat_train, class_train)
	predicted = clf.predict(feat_test)
	
	score = r2_score(class_test, predicted)
	scores = scores + score

print("average: ", scores/10)

print("beginning training/testing with rf...")

scores = 0

for train_index, test_index in skf.split(features_data, class_data):
	feat_train, feat_test = features_data[train_index], features_data[test_index]
	class_train, class_test = class_data[train_index], class_data[test_index]
	
	clf = RandomForestRegressor(max_depth=2, random_state=0)
	clf.fit(feat_train, class_train)
	predicted = clf.predict(feat_test)
	
	score = r2_score(class_test, predicted)
	scores = scores + score

print("average: ", scores/10)
	
print("beginning training/testing with svm...")

scores = 0

for train_index, test_index in skf.split(features_data, class_data):
	feat_train, feat_test = features_data[train_index], features_data[test_index]
	class_train, class_test = class_data[train_index], class_data[test_index]
	
	clf = svm.SVR()
	clf.fit(feat_train, class_train)
	predicted = clf.predict(feat_test)
	
	score = r2_score(class_test, predicted)
	scores = scores + score

print("average: ", scores/10)

print("beginning training/testing with mlp...")

scores = 0

for train_index, test_index in skf.split(features_data, class_data):
	feat_train, feat_test = features_data[train_index], features_data[test_index]
	class_train, class_test = class_data[train_index], class_data[test_index]
	
	clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1)
	clf.fit(feat_train, class_train)
	predicted = clf.predict(feat_test)
	
	score = r2_score(class_test, predicted)
	scores = scores + score

print("average: ", scores/10)