import math
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import numpy as np
from collections import defaultdict
import csv
def training(Data,tf_idf_s,dataCount,levelArr):
    total = len(Data)#总样本个数
    y_prob = {}
    tags = defaultdict(int)
    for level in levelArr:
        tags[level] = 0
    #计算标签数量

    for tag in Data:
        tags[tag]+=1
    tagsData = np.transpose([Data])
    #采用极大似然数估计计算p(y)
    for tag in tags:
        y_prob[tag]=float(tags[tag])/total
    # 个数除以总
    #计算条件概率
    c_prob=np.zeros([len(tags),len(tf_idf_s[0])])
    Z=np.zeros([len(tags),1])
    for blockid in range(len(tf_idf_s)):
        tid = list(tags.keys()).index(Data[blockid])
        c_prob[tid]+=tf_idf_s[blockid]
        Z[tid]=np.sum(c_prob[tid])
    # c_prob/=Z
    for i in range(len(Z)):
        if Z[i]!=0:
            c_prob[i]/=Z[i]
    # 每个类别 每个词的tfidi合 除以各自的总合
    result=np.zeros([len(dataCount),len(tags)])
    for i in range(len(dataCount)):
        vec=dataCount[i]
        scoreArr = []
        for level in levelArr:
            tid = list(tags.keys()).index(level)
            score = np.sum(vec * c_prob[tid] * y_prob[level])
            scoreArr.append(score)
        result[i] = scoreArr

        # for y,pc in zip(y_prob,c_prob):
        #     score=np.sum(vec*pc*y_prob[y])
    return result
def fileProcess(loadpath):
    count_vec = CountVectorizer()
    transformer = TfidfVectorizer()
    data = pd.read_csv(loadpath)
    textF = ["ExceptionType","MethodCallName","MethodCallerName","VariableDeclarationType","VariableDeclarationName","ClassName","PackageName"]
    levelArr = ['info', 'error', 'trace', 'debug', 'warn', 'fatal']
    data.columns = ["CheckType","BlockType","MaxLogLevel","AssertInBlock","ThreadInBlock","JDBCInBlock","LogInBlock","ReturnInBlock","ThrowInBlock","SettingFlag","BlockSLOC","LogInBlockCount","MethodCallCount","MethodParameterCount","VariableDeclarationCount","Logdensity","LogNumber","AverageLogLength" ,"AverageeLogParameterCount","ExceptionType","MethodCallName","MethodCallerName","VariableDeclarationType","VariableDeclarationName","ClassName","PackageName","LogLevel"]
    data = data.sample(frac=1)
    data = data.fillna(" ")
    otherData = data.drop(["ExceptionType","MethodCallName","MethodCallerName","VariableDeclarationType","VariableDeclarationName","ClassName","PackageName", "LogLevel"],
                          axis=1).values
    tageData = data["LogLevel"].values
    for f in textF:
        textData = data[f].values
        try :
            dataCount = count_vec.fit_transform(textData).toarray()
            dataTfidf = transformer.fit_transform(textData).toarray()
        except:
            totalResult = np.zeros([len(tageData),6])
        else:
            half = (int)(len(textData) / 2)
            tageSecondData = tageData[:half]
            tageFirstData = tageData[half:]
            dataFirstCount = dataCount[half:]
            dataFirstTfidf = dataTfidf[half:]
            dataSecondCount = dataCount[:half]
            dataSecondTfidf = dataTfidf[:half]
            try:
                secondResult = training(tageFirstData, dataFirstTfidf, dataSecondCount, levelArr)
                firstResult = training(tageSecondData, dataSecondTfidf, dataFirstCount, levelArr)

            except:
                print(loadpath)
            else:
                print("wu")

            checkResult(secondResult,tageSecondData,levelArr)
            checkResult(firstResult,tageFirstData, levelArr)
            # print("........")
            # for i in flag:
            #     print(secondResult[i])
            totalResult = np.vstack([firstResult, secondResult])
        otherData = np.hstack([otherData, totalResult])
        # try:
        #     count = count_vec.fit_transform(textData)
        # except:
        #     print(loadpath)
        # else:
        #     tfidf_matrix = transformer.fit_transform(count)
        #     c = count.toarray()
        #     t = tfidf_matrix.toarray()
        #     otherData = np.hstack([otherData,training(tageData, t, c)])
    otherData = np.hstack([otherData, np.reshape(tageData, (-1, 1))])
    print(otherData.shape)
    return  otherData
def checkResult(result,tage,levelArr):
    flag = []
    for i in range(len(result)):
        nums = result[i]
        maxN = max(nums)
        if maxN != 0:
            index = list(nums).index(maxN)
            if (levelArr[index] == tage[i]):
                newNums = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                newNums[index] = 1.0
                result[i] = newNums
                flag.append(i)
    return flag
if __name__ == '__main__':
    loadpath = "E:\\graduate-design\\gittest\\bazel.csv"
    path = "E:\graduate-design\gittest27"
    outpath = 'resultData4.csv'
    list_name = []
    resultData = []
    flag = 1
    if flag == 0:
        fileProcess(loadpath)
    else:
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.splitext(file_path)[1] == '.csv':
                list_name.append(file_path)
        for txtPath in list_name:
            resultData.append(fileProcess(txtPath))

        print(len(resultData))
        if os.path.exists(outpath):
            os.remove(outpath)
        with open(outpath, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, dialect='excel')
            for data in resultData:
                for line in enumerate(data):
                    # print(line[1])
                    # oneline = line[1].strip("\n").split(",")
                    csv_writer.writerow(line[1])

