import os,sys
import pandas as pd
import csv
import numpy as np
from collections import defaultdict
if __name__ == '__main__':
    path = "./resultDataTest1.csv"
    outpath = "Summary.csv"
    list_name = []
    resultData = []
    data = pd.read_csv(path)
    data.columns = ["CheckType", "BlockType", "MaxLogLevel", "AssertInBlock", "ThreadInBlock", "JDBCInBlock",
                    "LogInBlock", "ReturnInBlock", "ThrowInBlock", "SettingFlag", "BlockSLOC", "LogInBlockCount",
                    "MethodCallCount", "MethodParameterCount", "VariableDeclarationCount", "Logdensity", "LogNumber",
                    "AverageLogLength", "AverageeLogParameterCount", "ExceptionType1", "ExceptionType2",
                    "ExceptionType3", "ExceptionType4", "ExceptionType5", "ExceptionType6", "LoopCondition1",
                    "LoopCondition2", "LoopCondition3", "LoopCondition4", "LoopCondition5", "LoopCondition6",
                    "LogicBranchCondition1", "LogicBranchCondition2", "LogicBranchCondition3", "LogicBranchCondition4",
                    "LogicBranchCondition5", "LogicBranchCondition6", "MthodBlockType1", "MthodBlockType2",
                    "MthodBlockType3", "MthodBlockType4", "MthodBlockType5", "MthodBlockType6", "MethodcallName1",
                    "MethodcallName2", "MethodcallName3", "MethodcallName4", "MethodcallName5", "MethodcallName6",
                    "MethodCallerName1", "MethodCallerName2", "MethodCallerName3", "MethodCallerName4",
                    "MethodCallerName5", "MethodCallerName6", "VariableDeclarationType1", "VariableDeclarationType2",
                    "VariableDeclarationType3", "VariableDeclarationType4", "VariableDeclarationType5",
                    "VariableDeclarationType6",
                    "VariableDeclarationName1", "VariableDeclarationName2", "VariableDeclarationName3",
                    "VariableDeclarationName4", "VariableDeclarationName5", "VariableDeclarationName6", "ClassName1",
                    "ClassName2", "ClassName3", "ClassName4", "ClassName5", "ClassName6", "PackageName1",
                    "PackageName2", "PackageName3", "PackageName4", "PackageName5", "PackageName6", "LogLevel"]
    otherData = data.drop(["CheckType", "MaxLogLevel", "AssertInBlock", "ThreadInBlock", "JDBCInBlock",
                    "LogInBlock", "ReturnInBlock", "ThrowInBlock", "SettingFlag", "BlockSLOC", "LogInBlockCount",
                    "MethodCallCount", "MethodParameterCount", "VariableDeclarationCount", "Logdensity", "LogNumber",
                    "AverageLogLength", "AverageeLogParameterCount", "ExceptionType1", "ExceptionType2",
                    "ExceptionType3", "ExceptionType4", "ExceptionType5", "ExceptionType6", "LoopCondition1",
                    "LoopCondition2", "LoopCondition3", "LoopCondition4", "LoopCondition5", "LoopCondition6",
                    "LogicBranchCondition1", "LogicBranchCondition2", "LogicBranchCondition3", "LogicBranchCondition4",
                    "LogicBranchCondition5", "LogicBranchCondition6", "MthodBlockType1", "MthodBlockType2",
                    "MthodBlockType3", "MthodBlockType4", "MthodBlockType5", "MthodBlockType6", "MethodcallName1",
                    "MethodcallName2", "MethodcallName3", "MethodcallName4", "MethodcallName5", "MethodcallName6",
                    "MethodCallerName1", "MethodCallerName2", "MethodCallerName3", "MethodCallerName4",
                    "MethodCallerName5", "MethodCallerName6", "VariableDeclarationType1", "VariableDeclarationType2",
                    "VariableDeclarationType3", "VariableDeclarationType4", "VariableDeclarationType5",
                    "VariableDeclarationType6",
                    "VariableDeclarationName1", "VariableDeclarationName2", "VariableDeclarationName3",
                    "VariableDeclarationName4", "VariableDeclarationName5", "VariableDeclarationName6", "ClassName1",
                    "ClassName2", "ClassName3", "ClassName4", "ClassName5", "ClassName6", "PackageName1",
                    "PackageName2", "PackageName3", "PackageName4", "PackageName5", "PackageName6"],
                          axis=1).values
    sum = defaultdict(object)
    result = np.zeros([10, 6])
    blockType = ["IfBlock","ForBlock","DoBlock","WhileBlock","TryBlock","CatchBlock","SwitchBlock","MethodBlock","SynchronizedBlock","ClassBlock"]
    levelArr = ["", 'trace', 'debug','info', 'warn',  'error','fatal']
    # print(otherData)
    for b in blockType:
        for l in levelArr:
            sum[b]= {
                l:0
            }
    for vec in otherData:
        i = list(blockType).index(vec[0])
        j = list(levelArr).index(vec[1])-1
        result[i][j]+=1
    print(result)
    if os.path.exists(outpath):
        os.remove(outpath)
    with open(outpath, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, dialect='excel')
        csv_writer.writerow(levelArr)
        for i in range(len(result)):
            list = []
            list.append( blockType[i])
            for r in result[i]:
              list.append(r)
            csv_writer.writerow(list)