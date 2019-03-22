import os,sys
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
if __name__=="__main__"
    path = "E:\graduate-design\git"
    outpath = 'resultData.csv'
    list_name = []
    resultData = []
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