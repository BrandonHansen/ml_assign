from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

file_name = "regression_train.data"

print("collecting data for training...")

full_feature_cols = [i for i in range(0, 21)]
class_labels = 21

full_features = pd.read_csv(file_name, usecols=full_feature_cols)
classes = pd.read_csv(file_name, usecols=[class_labels])

selected_features = full_features

features_data = selected_features.as_matrix()
class_data = np.ravel(classes.as_matrix())

clf = RandomForestRegressor(max_depth=2, random_state=0)
clf.fit(features_data, class_data)

predicted = clf.predict(features_data)
score = r2_score(class_data, predicted)
print("resubstitution score:", score)

file_name = "regression_test.test"

print("collecting data for testing...")

full_feature_cols = [i for i in range(0, 21)]
class_labels = 21

full_features = pd.read_csv(file_name, usecols=full_feature_cols)
classes = pd.read_csv(file_name, usecols=[class_labels])

selected_features = full_features

features_data = selected_features.as_matrix()
class_data = np.ravel(classes.as_matrix())

predicted = clf.predict(features_data)
score = r2_score(class_data, predicted)
print("nonsense score:", score)



np.savetxt("regression_results.csv", predicted, delimiter=",")