from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

data_sample = "none"


file_name = "classification_train.txt"

print("collecting data for training...")

full_cols = [i for i in range(0, 49)]
full_feature_cols = [i for i in range(0, 48)]
cont_feature_cols = [3,14,24,39]
disc_feature_cols = [i for i in range(0, 48) if i not in cont_feature_cols]
class_labels = 48


full_features = pd.read_csv(file_name, usecols=full_cols, header=None)
'''
cont_features = pd.read_csv(file_name, usecols=cont_feature_cols)
disc_features = pd.read_csv(file_name, usecols=disc_feature_cols)
classes = pd.read_csv(file_name, usecols=[class_labels])
'''

###RESAMPLE
if data_sample == "down" or data_sample == "up":
	minority = full_features[full_features[class_labels] == 1]
	majority = full_features[full_features[class_labels] == -1]

	
	if data_sample = "down":
		###MAJORITY DOWNSAMPLE
		maj_downsampled = resample(majority, replace=True, n_samples=1029)

		downsampled = pd.concat([majority, min_downsampled])

		resampled = downsampled
	else:
		###MINORITY UPSAMPLEs
		min_upsampled = resample(minority, replace=True, n_samples=3118)

		upsampled = pd.concat([majority, min_upsampled])

		resampled = upsampled

		resampled = resampled.sample(frac=1).reset_index(drop=True)

		cont_features = resampled[cont_feature_cols]
		disc_features = resampled[disc_feature_cols]
		classes = resampled[class_labels]
else:
	cont_features = pd.read_csv(file_name, usecols=cont_feature_cols)
	disc_features = pd.read_csv(file_name, usecols=disc_feature_cols)
	classes = pd.read_csv(file_name, usecols=[class_labels])

selected_features = cont_features

features_data = selected_features.as_matrix()
class_data = np.ravel(classes.as_matrix())

clf = svm.SVC()
clf.fit(features_data, class_data)

features_data = pd.read_csv(file_name, usecols=cont_feature_cols).as_matrix()
class_data = np.ravel(pd.read_csv(file_name, usecols=[class_labels]).as_matrix())
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