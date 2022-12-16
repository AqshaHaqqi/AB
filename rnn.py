# multi-class classification with Keras
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# load dataset
dataframe = pandas.read_csv("dataSetaqsha.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:1000].astype(float)
Y = dataset[:,1000]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# create model
model = Sequential()
model.add(Dense(8, input_dim=1000, activation='relu'))
model.add(Dense(6, activation='softmax'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

nepochs = 200
nbatch = 5

model.fit(X, dummy_y, epochs=nepochs, batch_size=nbatch)
_, accuracy = model.evaluate(X, dummy_y)
print('Accuracy: %.2f' % (accuracy*100))
