import numpy as np
import urllib.request
loadpath = "E:\graduate-design\test-pro\py-test\2.csv"
# load the CSV file as a numpy matrix
dataset = np.loadtxt(loadpath, delimiter=",")
# separate the data from the target attributes
X = dataset[:,0:15]
y = dataset[:,15]
print("size:",dataset.size)
