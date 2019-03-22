from __future__ import print_function
import math
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, LabelBinarizer

from sklearn.model_selection import train_test_split, GridSearchCV
def next_batch(X, Y,batch_i, batch_size):
    start = batch_i * batch_size
    end = start + batch_size
    batch_xs = X[start:end, :]
    batch_ys = Y[start:end, :]
    return batch_xs, batch_ys  # 生成每一个batch
def generatebatch(X,Y,n_examples, batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i*batch_size
        end = start + batch_size
        batch_xs = X[start:end, :]
        batch_ys = Y[start:end, :]
        yield batch_xs, batch_ys # 生成每一个batch

if __name__ == '__main__':
    print('Start read data')
    learning_rate = 0.1
    training_epochs = 1000
    batch_size = 100
    display_step = 50
    n_input = 25
    n_classes = 6
    loadpath = "E:\\graduate-design\\test-pro\py-test\\resultData1.csv"
    encoder = LabelEncoder()
    one_hot = OneHotEncoder(categories='auto')
    data = pd.read_csv(loadpath)

    # sess = tf.InteractiveSession()
    data = data.sample(frac=1)
    # # Y = data[[42]]
    # mianData = data[list(range(18))]
    # textData = data[list(range(18,42))]
    # # print(Y[:2])
    # print(mianData[:2])
    # print(textData[:2])
    # data.columns = ["CheckType","BlockType","MaxLogLevel","AssertInBlock","ThreadInBlock","JDBCInBlock","LogInBlock","ReturnInBlock","ThrowInBlock","SettingFlag","BlockSLOC","LogInBlockCount","MethodCallCount","MethodParameterCount","VariableDeclarationCount","Logdensity","LogNumber","AverageLogLength" ,"AverageeLogParameterCount", "ExceptionType1", "ExceptionType2",
    #                 "ExceptionType3", "ExceptionType4", "ExceptionType5", "ExceptionType6", "MethodcallName1",
    #                 "MethodcallName2", "MethodcallName3", "MethodcallName4", "MethodcallName5", "MethodcallName6",
    #                 "VariableDeclarationType1", "VariableDeclarationType2",
    #                 "VariableDeclarationType3", "VariableDeclarationType4", "VariableDeclarationType5",
    #                 "VariableDeclarationType6",
    #                 "VariableDeclarationName1", "VariableDeclarationName2", "VariableDeclarationName3",
    #                 "VariableDeclarationName4", "VariableDeclarationName5", "VariableDeclarationName6", "LogLevel"]

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
    # Y = encoder.fit_transform(Y)
    Y = np.reshape(Y, (-1, 1))
    Y = one_hot.fit_transform(Y).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    print(Y.shape)
    x = tf.placeholder("float", [None, X.shape[1]])
    y = tf.placeholder("float", [None, Y.shape[1]])
    w1 = tf.Variable(tf.random_normal([X.shape[1], Y.shape[1]]))
    b1 = tf.Variable(tf.random_normal([Y.shape[1]]))
    pred = tf.add(tf.matmul(x, w1), b1)
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(X_train.shape[0] / batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = next_batch(X_train, y_train, i, batch_size)
            # for batch_xs, batch_ys in generatebatch(X_train, y_train, Y.shape[0], batch_size):
                _, c = sess.run([optimizer, cross_entropy_loss], feed_dict={x: batch_xs, y: batch_ys})
                avg_cost += c / total_batch
            if epoch % display_step == 0:
                print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print ("Optimization Finished!")

        print ("Get test data:")
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print('调用函数auc：', metrics.roc_auc_score(y_core, y_acu, average='micro'))
        # 2、手动计算micro类型的AUC
        # 首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
        fpr, tpr, thresholds = metrics.roc_curve(y_core.ravel(), y_acu.ravel())
        auc = metrics.auc(fpr, tpr)
        print('手动计算auc：', auc)
        # 绘图
        mpl.rcParams['font.sans-serif'] = u'SimHei'
        mpl.rcParams['axes.unicode_minus'] = False
        # FPR就是横坐标,TPR就是纵坐标
        plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc)
        plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.grid(b=True, ls=':')
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
        plt.title(u'鸢尾花数据Logistic分类后的ROC和AUC', fontsize=17)
        plt.show()

        print ("Accuracy:", accuracy.eval({x: X_test, y:y_test}))


