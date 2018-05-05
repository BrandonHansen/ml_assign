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


#TRAIN TEST VARIABLES

#none, down, up
data_sample = "up"
#all, disc, cont
data_type = "cont"

print("experiement:", data_sample, data_type)

###DEFINE FEATURES AND CLASSES, GET DATA FROM FILE

file_name = "classification_train.txt"

print("collecting data...")

#DEFINE FEATURES AND CLASSES
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
		
	resampled = resampled.sample(frac=1).reset_index(drop=True)
		
	all_features = resampled[full_feature_cols]
	cont_features = resampled[cont_feature_cols]
	disc_features = resampled[disc_feature_cols]
	classes = resampled[class_labels]
else:
	#GET ORIGINAL PROPORTION
	all_features = pd.read_csv(file_name, usecols=full_feature_cols)
	cont_features = pd.read_csv(file_name, usecols=cont_feature_cols)
	disc_features = pd.read_csv(file_name, usecols=disc_feature_cols)
	classes = pd.read_csv(file_name, usecols=[class_labels])

#CHOOSE FEATURE TYPE
if data_type == "disc":
	selected_features = disc_features
elif data_type == "cont":
	selected_features = cont_features
else:
	selected_features = all_features

#TRANSFORM FEATURES AND CLASS TYPE
features_data = selected_features.as_matrix()
class_data = np.ravel(classes.as_matrix())


###DO TESTS

model = GaussianNB()
classifierTesting('bayes', model, features_data, class_data)
model = KNeighborsClassifier()
classifierTesting('knn', model, features_data, class_data)

model = RandomForestClassifier()
classifierTesting('rf', model, features_data, class_data)

model = LogisticRegression()
classifierTesting('logistic', model, features_data, class_data)
model = svm.SVC()
classifierTesting('svm', model, features_data, class_data)
model = MLPClassifier()
classifierTesting('mlp', model, features_data, class_data)
