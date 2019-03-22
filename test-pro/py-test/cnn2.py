import math
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import tree

from sklearn.metrics import f1_score
# from sklearn.grid_search import GridSearchCV
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.optimizers import SGD
from keras.utils import np_utils
if __name__ == '__main__':
    print('Start read data')
   # loadpath = "./resultData.csv"
    loadpath = "E:\\graduate-design\\test-pro\py-test\\resultData1.csv"
    encoder = LabelEncoder()
    one_hot = OneHotEncoder(categories='auto')
    data = pd.read_csv(loadpath)
    data = data.sample(frac=1)
    data.columns = ["CheckType", "BlockType", "MaxLogLevel", "AssertInBlock", "ThreadInBlock", "JDBCInBlock",
                    "LogInBlock", "ReturnInBlock", "ThrowInBlock", "SettingFlag", "BlockSLOC", "LogInBlockCount",
                    "MethodCallCount", "MethodParameterCount", "VariableDeclarationCount", "Logdensity", "LogNumber",
                    "AverageLogLength", "AverageeLogParameterCount", "ExceptionType1", "ExceptionType2",
                    "ExceptionType3", "ExceptionType4", "ExceptionType5", "ExceptionType6", "MethodcallName1",
                    "MethodcallName2", "MethodcallName3", "MethodcallName4", "MethodcallName5", "MethodcallName6",
                    "MethodCallerName1", "MethodCallerName2", "MethodCallerName3", "MethodCallerName4",
                    "MethodCallerName5", "MethodCallerName6", "VariableDeclarationType1", "VariableDeclarationType2",
                    "VariableDeclarationType3", "VariableDeclarationType4", "VariableDeclarationType5",
                    "VariableDeclarationType6",
                    "VariableDeclarationName1", "VariableDeclarationName2", "VariableDeclarationName3",
                    "VariableDeclarationName4", "VariableDeclarationName5", "VariableDeclarationName6", "ClassName1",
                    "ClassName2", "ClassName3", "ClassName4", "ClassName5", "ClassName6", "PackageName1",
                    "PackageName2", "PackageName3", "PackageName4", "PackageName5", "PackageName6", "LogLevel"]
    numeric_features = ["BlockSLOC", "LogInBlockCount", "MethodCallCount", "MethodParameterCount",
                        "VariableDeclarationCount", "Logdensity", "LogNumber", "AverageLogLength",
                        "AverageeLogParameterCount"]
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_features = ["CheckType", "BlockType", "MaxLogLevel"]
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)], remainder='passthrough')
    # clf = Pipeline(steps=[('preprocessor', preprocessor)])
    X = data.drop("LogLevel", axis=1)
    X = preprocessor.fit_transform(X)
    X = np.reshape(X, (X.shape[0], -1)).astype(np.float32)
    Y = data["LogLevel"].values
    Y = encoder.fit_transform(Y)
    Y = np_utils.to_categorical(Y, 6)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    #Y = Y.reshape((Y.shape[0], Y.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)
    model = Sequential()
    model.add(Convolution1D(nb_filter=512, filter_length=1))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(6))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    nb_epoch = 15
    model.fit(X_train, y_train, nb_epoch=nb_epoch, validation_data=(X_test, y_test), batch_size=16)

