from __future__ import print_function
import math
import pandas as pd
import numpy as np
import random
import time

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
def next_batch(train_data, train_target, batch_size):
    index = [i for i in range(0,len(train_target))]
    np.random.shuffle(index)
    batch_data = []
    batch_target = []
    for i in range(0, batch_size):
        batch_data.append(train_data[index[i]])
        batch_target.append(train_target[index[i]])
    return batch_data, batch_target

if __name__ == '__main__':
    print('Start read data')
    learning_rate = 0.01
    training_epochs = 1000
    batch_size = 100
    display_step = 50
    n_input = 28
    n_classes = 6
    loadpath = "E:\\graduate-design\\test-pro\py-test\\test.csv"
    encoder = LabelEncoder()
    one_hot = OneHotEncoder(categories='auto')
    data = pd.read_csv(loadpath)
    data.columns = ["CheckType", "BlockType", "BlockSLOC", "ExceptionRatio", "ReturnInBlock", "ThrowInBlock",
                    "SettingFlag", "MethodCallCount", "MethodParameterCount", "VariableDeclarationCount", "Logdensity",
                    "LogNumber", "AverageLogLength", "AverageeLogParameterCount", "LogLevel"]
    # sess = tf.InteractiveSession()
    numeric_features = ["BlockSLOC", "MethodCallCount", "MethodParameterCount", "VariableDeclarationCount", "LogNumber",
                        "AverageLogLength", "AverageeLogParameterCount"]
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_features = ["CheckType", "BlockType"]
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
    # Y = encoder.fit_transform(Y)
    Y = np.reshape(Y, (-1, 1))
    Y = one_hot.fit_transform(Y).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    x = tf.placeholder("float", [None, n_input])
    y_ = tf.placeholder("float", [None, n_classes])
    print(X.shape)
    print(Y.shape)
    eval_sklearn = False
    if eval_sklearn:
        print
        "Start evaluating softmax regression model by sklearn..."
        reg = LogisticRegression(solver="lbfgs", multi_class="multinomial")
        reg.fit(X_train, y_train)
        np.savetxt('coef_softmax_sklearn.txt', reg.coef_, fmt='%.6f')  # Save coefficients to a text file
        test_y_predict = reg.predict(X_test)
        print
        "Accuracy of test set: %f" % accuracy_score(y_test, test_y_predict)
    eval_tensorflow = True
    batch_gradient = False
    if eval_tensorflow:
        print("Start evaluating softmax regression model by tensorflow...")
        w = tf.Variable(tf.random_normal([n_input, n_classes]))
        b = tf.Variable(tf.random_normal([n_classes]))
        y = tf.nn.softmax(tf.matmul(x, w) + b)
        y_ = tf.placeholder(tf.float32, [None, 6])


        loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)

        if batch_gradient:
            for step in range(300):
                sess.run(train, feed_dict={x: X_train, y_: y_test})
                if step % 10 == 0:
                    print
                    "Batch Gradient Descent processing step %d" % step
            print
            "Finally we got the estimated results, take such a long time..."
        else:
            for step in range(1000):
                batch_xs, batch_ys = next_batch(X_train, y_train, 100)
                sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
                if step % 100 == 0:
                    print ("Stochastic Gradient Descent processing step %d" % step)
        np.savetxt('coef_softmax_tf.txt', np.transpose(sess.run(w)), fmt='%.6f')  # Save coefficients to a text file
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy of test set: %f" % sess.run(accuracy, feed_dict={x: X_test, y_: y_test}))
