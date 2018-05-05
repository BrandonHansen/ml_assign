from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

def regressorTesting(name, model, features_data, class_data):

	print("beginning training/testing with "+name+"...")

	scores = 0

	for count in range(0, 10):
		feat_train, feat_test, class_train, class_test = train_test_split(features_data, class_data, test_size=0.33)
		
		clf = model
		clf.fit(feat_train, class_train)
		predicted = clf.predict(feat_test)
		
		score = r2_score(class_test, predicted)
		scores = scores + score

	print("r2: ", scores/10)
	
def classifierTesting(name, model, features_data, class_data):

	skf = StratifiedKFold(n_splits=10)

	print("beginning training/testing with "+name+"...")

	scores = 0
	fscores = 0

	for train_index, test_index in skf.split(features_data, class_data):
		feat_train, feat_test = features_data[train_index], features_data[test_index]
		class_train, class_test = class_data[train_index], class_data[test_index]
		
		clf = model
		clf.fit(feat_train, class_train)
		predicted = clf.predict(feat_test)
		
		score = accuracy_score(class_test, predicted)
		scores = scores + score
		
		fscore = f1_score(class_test, predicted, average='macro')  
		fscores = fscores + fscore
		
		#print("subscore:", score, "subf1:", fscore)


	print("accuracy: ", scores/10, ", f1: ", fscores/10)