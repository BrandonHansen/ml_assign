from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from testing_functions import classifierTesting
import pandas as pd
import numpy as np



data_sample = "none"

###DEFINE FEATURES AND CLASSES, GET DATA FROM FILE

file_name = "classification_train.txt"

print("collecting data...")

full_cols = [i for i in range(0, 49)]
full_feature_cols = [i for i in range(0, 48)]
cont_feature_cols = [3,14,24,39]
disc_feature_cols = [i for i in range(0, 48) if i not in cont_feature_cols]
class_labels = 48

full_features = pd.read_csv(file_name, usecols=full_cols, header=None)

if data_sample == "down" or data_sample == "up":
	###BALANCE SAMPLING FOR BETTER PREDICTIONS

	minority = full_features[full_features[class_labels] == 1]
	majority = full_features[full_features[class_labels] == -1]
	if data_sample == "down":
		###MAJORITY DOWNSAMPLE
		maj_downsampled = resample(majority, replace=True, n_samples=1029)

		downsampled = pd.concat([minority, maj_downsampled])

		resampled = downsampled
	else:
		###MINORITY UP SAMPLE
		min_upsampled = resample(minority, replace=True, n_samples=3118)

		upsampled = pd.concat([majority, min_upsampled])

		resampled = upsampled
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


###DO TESTS
'''
model = GaussianNB()
classifierTesting('bayes', model, features_data, class_data)
model = KNeighborsClassifier(n_neighbors=3)
classifierTesting('knn', model, features_data, class_data)
model = RandomForestClassifier(max_depth=2, random_state=0)
classifierTesting('rf', model, features_data, class_data)
model = LogisticRegression(C=1e5)
classifierTesting('logistic', model, features_data, class_data)
'''
model = svm.SVC(kernel='linear')
classifierTesting('svm', model, features_data, class_data)
model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 50), random_state=1)
classifierTesting('mlp', model, features_data, class_data)
