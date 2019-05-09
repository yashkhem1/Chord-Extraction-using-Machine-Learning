import numpy as np
import pickle
from utilities import CtoN
import pandas as pd
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import RBFSampler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# model = SGDClassifier(loss='hinge', learning_rate='constant',
#                       eta0=0.1, max_iter=2000, tol=1e-3)

# For trained_ML_model_ver1 we have used MLP classifier uncomment it below and comment
# SGD classifier to use NN based methods for training, thought MLPclassifier
# provides no opportunity for further training

model = MLPClassifier(solver='adam',activation='logistic',
                      alpha=1e-2, hidden_layer_sizes=(35,),
                      momentum=0.25, random_state=1)

print(model)

# load the existing dataset with PCP vectors in rows
data_set = pd.read_csv("PCP_train_data.csv")
dim = data_set.shape
x = dim[0]
y = dim[1] - 1
dim = (x, y)
X = np.zeros(dim)
i = 0
while i < 12:
    X[:, i] = data_set[str(i)]
    i += 1
# Manually creating label values according to data per chord
# It is assumed that the chords are listed in the order
# A, Am, Bm, C, D, Dm, E, Em, F, G in the dataset
y = np.zeros((X.shape)[0])
counter = 0
value = 1
data_per_chord = 200
for i in range(0, (X.shape)[0]):
    if counter == data_per_chord:
        value += 1
        counter = 0
    y[i] = value
    counter += 1
sampler = AdditiveChi2Sampler()
# Comment the above sampler and uncomment the lower one to change kernels

#sampler = RBFSampler(gamma=1, random_state=1)

p = np.random.permutation(len(X))
X = X[p]
y = y[p]
X = sampler.fit_transform(X)
X_train = X[0:1500]
y_train = y[0:1500]
X_test = X[1500:]
y_test = y[1500:]
model.fit(X_train, y_train)
filename = 'trained_ML_model_ver3.sav'
# Fit and save the model with filename
pickle.dump(model, open(filename, 'wb'))
# Load back the model to test for training accuracy
myModel = pickle.load(open('trained_ML_model_ver3.sav', 'rb'))
pred = myModel.predict(X_test)
print(accuracy_score(pred, y_test))
