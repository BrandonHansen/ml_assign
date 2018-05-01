import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_name = "classification_train.txt"

print("collecting data for classification...")

class_labels = 48

classes = pd.read_csv(file_name, usecols=[class_labels])

sum_pos = classes[classes == 1].count()
sum_neg = classes[classes == -1].count()

total = sum_pos[0]+sum_neg[0]
print("pos fraction:", sum_pos[0]/total)
print("neg fraction:", sum_neg[0]/total)

file_name = "regression_train.data"

print("collecting data for regression...")

class_labels = 21

classes = pd.read_csv(file_name, usecols=[class_labels])

classes.plot.hist(alpha=0.5, bins=20)

plt.show()