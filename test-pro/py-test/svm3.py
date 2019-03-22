from __future__ import print_function
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
from sklearn.svm import SVC
from sklearn.metrics import f1_score
# from sklearn.grid_search import GridSearchCV
if __name__ == '__main__':
    print('Start read data')
    loadpath = "/root/Code/resultData.csv"
    encoder = LabelEncoder()
    one_hot = OneHotEncoder(categories='auto')
    data = pd.read_csv(loadpath)
    data = data.sample(frac=1)
    data.columns = ["CheckType","BlockType","MaxLogLevel","AssertInBlock","ThreadInBlock","JDBCInBlock","LogInBlock","ReturnInBlock","ThrowInBlock","SettingFlag","BlockSLOC","LogInBlockCount","MethodCallCount","MethodParameterCount","VariableDeclarationCount","Logdensity","LogNumber","AverageLogLength" ,"AverageeLogParameterCount", "ExceptionType1", "ExceptionType2","ExceptionType3","ExceptionType4","ExceptionType5","ExceptionType6","MethodcallName1","MethodcallName2","MethodcallName3","MethodcallName4","MethodcallName5","MethodcallName6", "MethodCallerName1","MethodCallerName2","MethodCallerName3","MethodCallerName4","MethodCallerName5","MethodCallerName6","VariableDeclarationType1","VariableDeclarationType2","VariableDeclarationType3","VariableDeclarationType4","VariableDeclarationType5","VariableDeclarationType6",
                  "VariableDeclarationName1","VariableDeclarationName2","VariableDeclarationName3","VariableDeclarationName4","VariableDeclarationName5","VariableDeclarationName6","ClassName1","ClassName2","ClassName3","ClassName4","ClassName5","ClassName6","PackageName1","PackageName2","PackageName3","PackageName4","PackageName5","PackageName6","LogLevel"]
    numeric_features = ["BlockSLOC","LogInBlockCount","MethodCallCount","MethodParameterCount","VariableDeclarationCount","Logdensity","LogNumber","AverageLogLength" ,"AverageeLogParameterCount"]
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_features = ["CheckType", "BlockType", "MaxLogLevel"]
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)], remainder='passthrough')
    # clf = Pipeline(steps=[('preprocessor', preprocessor)])
    X = data.drop("LogLevel", axis=1)
    print(X[:1])
    X = preprocessor.fit_transform(X)
    X = np.reshape(X, (X.shape[0], -1)).astype(np.float32)
    Y = data["LogLevel"].values
    Y = encoder.fit_transform(Y)
    random_state = np.random.RandomState(0)
    Y = np.reshape(Y, (-1, 1))
    # Y = one_hot.fit_transform(Y).toarray()
    # Y = label_binarize(Y, classes=list(range(6)))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    print(X.shape)
    # model = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
    model = SVC(decision_function_shape="ovr",probability=True)
    # model = OneVsRestClassifier(svm.SVC(kernel="poly"))
    # parameters = {
    #     "estimator__C": [1, 2, 4, 8],
    #     "estimator__kernel": ["poly", "rbf"],
    #     "estimator__degree": [1, 2, 3, 4],
    # }
    # model_tunning = GridSea      rchCV(model, param_grid=parameters)
    print("start fit")
    clf = model.fit(X_train, y_train)
    print("end fit")
    y_ = clf.predict(X_test)
    print('Accracy:', clf.score(X_test, y_test))
    # print(model_tunning.best_score_)
    # print(model_tunning.best_params_)
    # y_score = one_hot.fit_transform(y_test).toarray()
    # y_ = clf.predict(X_test)
    # y_test = np.reshape(y_test, (-1, 1))
    # y_= np.reshape(y_, (-1, 1))
    # len = y_test.shape[0]
    # # totalY = np.vstack([y_core, y_core])
    # # totalY = one_hot.fit_transform(totalY).toarray()
    # # print(totalY)
    # y_core = one_hot.fit_transform(y_test).toarray()
    # y_acu = one_hot.fit_transform(y_).toarray()
    # # y_ = one_hot.fit_transform(y_).toarray()
    y_p = clf.predict_proba(X_test)
    brier_score = 0
    for y, yp in zip(y_test, y_p):
        for i in range(y_p.shape[1]):
            brier_score += (yp[i] - y[i]) * (yp[i] - y[i])

    print("brier_score:", brier_score / y_p.shape[0])
    print(y_test[:2])
    print(y_[:2])
    print('调用函数auc：', metrics.roc_auc_score(y_test, y_, average='micro'))
    # 2、手动计算micro类型的AUC
    # 首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
    fpr, tpr, thresholds = metrics.roc_curve(y_test.ravel(), y_.ravel())
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
    #
    # clf_score = brier_score_loss(y_test, y_)
    # print("Brier score: %1.3f" % clf_score)

