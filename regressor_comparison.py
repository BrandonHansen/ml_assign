from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from testing_functions import regressorTesting
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

model = KNeighborsRegressor(n_neighbors=3)
regressorTesting('knn', model, features_data, class_data)
model = RandomForestRegressor(max_depth=2, random_state=0)
regressorTesting('rf', model, features_data, class_data)
model = svm.SVR()
regressorTesting('svm', model, features_data, class_data)
model = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 10), random_state=1)
regressorTesting('mlp', model, features_data, class_data)
