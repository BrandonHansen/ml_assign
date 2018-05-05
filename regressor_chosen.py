from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

###DEFINE FEATURES AND CLASSES, GET DATA FROM FILE

file_name = "regression_train.data"

print("collecting data for training...")

#DEFINE FEATURES AND CLASSES
full_feature_cols = [i for i in range(0, 21)]
class_labels = 21

#GET FROM FILE
full_features = pd.read_csv(file_name, usecols=full_feature_cols)
classes = pd.read_csv(file_name, usecols=[class_labels])

#SELECT FEATURES
selected_features = full_features

#TRANSFORM FEATURES AND CLASS TYPE
features_data = selected_features.as_matrix()
class_data = np.ravel(classes.as_matrix())

#TRAIN ON FULL DATA
clf = RandomForestRegressor()
clf.fit(features_data, class_data)

#TEST SUCCESS BY RESUBSTITUTION
predicted = clf.predict(features_data)
score = r2_score(class_data, predicted)
print("resubstitution score:", score)

#GET TEST DATA
file_name = "regression_test.test"

print("collecting data for testing...")

#DEFINE FEATURES
full_feature_cols = [i for i in range(0, 21)]
class_labels = 21

#GET FROM TEST FILE
full_features = pd.read_csv(file_name, usecols=full_feature_cols)
classes = pd.read_csv(file_name, usecols=[class_labels])

selected_features = full_features

#TRANSFORM FEATURES AND CLASS TYPE
features_data = selected_features.as_matrix()
class_data = np.ravel(classes.as_matrix())

#MAKE PREDICTION
predicted = clf.predict(features_data)
score = r2_score(class_data, predicted)
print("nonsense score:", score)

#WRITE TO NEW FILE
predicted_formatted = np.ravel(predicted)

write_file = open('regression_results.csv', 'w')

for prediction in predicted_formatted:
	write_file.write("%s\n" % prediction)
  
write_file.close()