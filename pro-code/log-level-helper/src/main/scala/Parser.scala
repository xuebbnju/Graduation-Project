
import java.io.{File, PrintWriter}
import java.util.ArrayList

import Parser.generate
import scala.math._
import scala.collection.mutable
import scala.io.Source

/**
  * Author: xbb
  * Date: 18/12/18
  */
object Parser {
  val root = "E:\\graduate-design\\"
 val name = "git-project" //"git-project\storm\storm-core\src\jvm"
  val featureArr = Array("CheckType","BlockType","MaxLogLevel","AssertInBlock","ThreadInBlock","JDBCInBlock","LogInBlock","ReturnInBlock","ThrowInBlock","SettingFlag","BlockSLOC","LogInBlockCount","MethodCallCount","MethodParameterCount","VariableDeclarationCount","Logdensity","LogNumber","AverageLogLength" ,"AverageeLogParameterCount")
  val namet = "git-project\\elasticsearch"
  var logInfo: Vector[String] = Vector()
 //val featureArr = Array("CheckType","BlockType","ExceptionRatio","ReturnInBlock","ThrowInBlock","SettingFlag","MethodCallCount","MethodParameterCount","VariableDeclarationCount","Logdensity","LogNumber")
  def main(args: Array[String]): Unit = {
    val flag = 1
    val outFile = root+"git-outshao"
    val outputPath = root + "git-outshao"
    if(flag == 0){
      val projectPath = root + namet
      val source = new File(projectPath)
      val files = getJavaFiles(source).toVector//过滤掉不是blame的文件，且递归扁平化处理
      generate(parseProject(files,source.getName), outputPath+"\\"+source.getName+".txt")
      generate(logInfo,root+"git\\"+source.getName+".txt")
    }else {
      val projectPath = root + name
      val source = new File(projectPath)
      source.listFiles.foreach(f=>{
        if (f.isDirectory) { //判断是否为文件
          val files = getJavaFiles(f).toVector
          println("项目名称："+f.getName)
          generate(parseProject(files,f.getName),outFile+"\\"+f.getName+".txt")

        }

      })
      generate(logInfo,root+"git\\summaryshan.txt")
    }



  }

  def getJavaFiles(file: File): Array[File] = {
    val files = file.listFiles
    files.filter(_.isFile).filter(_.toString.endsWith(".blame")) ++ files.filter(_.isDirectory).flatMap(getJavaFiles)
  }

  def parseProject(files: Vector[File],proName: String): Vector[String] = {
    val textList = Array("ExceptionType","LoopCondition","LogicBranchCondition","MthodBlockType","MethodCallName","MethodCallerName","VariableDeclarationType","VariableDeclarationName","ClassName","PackageName")
    val fileInfo:mutable.LinkedHashMap[String, mutable.HashMap[String, String]] = mutable.LinkedHashMap()
    //val finalInfo:mutable.LinkedHashMap[String, mutable.HashMap[String, String]] = mutable.LinkedHashMap()
      var infoMap = files.map(file =>FileParser.parseFile(file,fileInfo)).reduce(_++_)
      // Use and close file
    //val featureInfo: mutable.LinkedHashMap[String, mutable.LinkedHashMap[String, String]] = mutable.LinkedHashMap()
    var result: Vector[String] = Vector()
    var textResult: Vector[String] = Vector()
    var exceptionMap =mutable.HashMap[String, Int]()
    var fileOwnership =mutable.HashMap[String, Int]()
    var exceptionSum = 0
    var blockTypeSet = mutable.Set[String]()
    var checkTypeSet = mutable.Set[String]()
    val SLOCInfo = getSLOCInfo(fileInfo)
    val mediaSloc = SLOCInfo("MediaSloc")
    val sumSLOC = SLOCInfo("SumSloc")
    val headAuthor = getHeadAuthors(proName)
    val fileNumThreshold = fileInfo.size * 0.01
    val logDensityThreshold = 0.01
    val largeFileThreshold = mediaSloc * 4
    println("fileInfo:")
    println(fileInfo)
//    println("headAuthor:"+headAuthor)
//    println("fileNumThreshold:"+fileNumThreshold)
//    println("largeFileThreshold:"+largeFileThreshold)

    fileInfo.foreach{case (index, list) =>{
      val authorArr = list("Authors").split(";")
      authorArr.foreach(f=>{
        if(fileOwnership.contains(f)) {
          fileOwnership += (f -> (fileOwnership(f) + 1))
        } else{
          fileOwnership += (f -> 1)
        }
      })
    }}
    println("fileOwnership:"+fileOwnership)
    var logDele = 0
    var levelDele =  0
    val levelPattern = "fatal|error|warn|debug|trace|config|fine|finer|finest|severe".r
    infoMap.foreach{case (index,list)=>{
        var flag = true
        val author = list("Author")
        val blockType = list("BlockType")
        val logLevel = list("LogLevel")
        val logArg = list("LogArg")
          levelPattern.findAllIn(logArg).foreach(level =>{
            val transLevel = level  match {
              case "error"|"severe" => "error"
              case "finest"|"trace" => "trace"
              case "fine"|"finer"|"debug" => "debug"
              case "info"|"config" => "info"
              case "warn" => "warn"
              case "fatal"=> "fatal"
            }
            if( flag && transLevel != logLevel){
              flag = false
            }
          })
        if(!flag){
          list += ("Flag"->"false")
          levelDele = levelDele+1
        }else if(blockType == "CatchBlock" && logLevel == "trace"){
          list += ("Flag"->"false")
          levelDele = levelDele+1
        }else if((blockType == "ForBlock" || blockType == "DoBlock"|| blockType == "WhileBlock" || blockType == "TryBlock")&& logLevel == "fatal"){
          list += ("Flag"->"false")
          levelDele = levelDele+1
        }else if(!headAuthor.contains(author)){
          val fileName = list("FileName")
          try {
            fileInfo(fileName)
            // Use and close file
          } catch {
            case ex: NoSuchElementException => {
              println(ex)
              println(fileName)
              println(list)
              //println(infoMap)
              }
          }
          val fileLogDesnity = fileInfo(fileName)("Logdensity").toDouble
          val fileSLOC = fileInfo(fileName)("SLOC").toInt
          try {
            fileOwnership(author)
            // Use and close file
          } catch {
            case ex: NoSuchElementException => {
              println(ex)
              println(author)
              println(list)
              //println(infoMap)
            }
          }
          if(fileOwnership(author)<fileNumThreshold || (fileSLOC > largeFileThreshold && fileLogDesnity < logDensityThreshold)){
            list += ("Flag"->"false")
            logDele=logDele+1
            //println("删除de日质数目："+list)
          }else{
            list += ("Flag"->"true")
          }
        }else{
          list += ("Flag"->"true")
        }
    }}
//    infoMap.foreach { case (index, list) =>
//    {
//      blockTypeSet += list.get("BlockType").get
//      checkTypeSet += list.get("CheckType").get
//      val exceptionType = list.get("ExceptionType").get
//      if(exceptionType != "" ){
//        exceptionSum += 1
//        if(exceptionMap.contains(exceptionType))
//          exceptionMap += (exceptionType -> (exceptionMap(exceptionType) + 1))
//        else
//          exceptionMap += (exceptionType -> 1)
//      }
//
//    }}
//    val logNum =  exceptionSum.toDouble
    infoMap.foreach { case (index, list) =>
    {
//      val exceptionType = list.get("ExceptionType").get
//      var exceptionRatio = "0"
//      if(exceptionType != "" ){
//        exceptionRatio = (exceptionMap.get(exceptionType).get.toDouble/logNum).formatted("%.2f")
//      }
//      list+=("ExceptionRatio" -> exceptionRatio)
      if(list("Flag") == "true"){
        var featureStr = ""
        for(c <- featureArr ){
          featureStr += list.get(c).get+","
        }
        for(c <- textList){
          try {
            featureStr += list.get(c).get.replaceAll("\\r|\\n|\\p{P}|\\n|\\+||\\=|,|\\<|\\>\\|", "")+","
            // Use and close file
          } catch {
            case ex: NoSuchElementException => {
              println(ex)
              println(c)
              println(list)
              //println(infoMap)
            }
          }

        }
        featureStr += list.get("LogLevel").get
        result = result :+ featureStr
      }
    }}
    println("总数SLOC数目："+sumSLOC)
    println("删除前日质数目："+infoMap.size)
    println(" levelDele:"+ levelDele)
    println(" logDele:"+ logDele)
    println("删除后日质数目："+result.size)
    val logNumInfo = proName + ","+sumSLOC+"," + infoMap.size +","+logDele+","+levelDele+","+result.size
    logInfo = logInfo :+ logNumInfo
//    generateTxt(textResult, root + name + "textF.txt")
    result
  }
  def getSLOCInfo(fileInfo: mutable.LinkedHashMap[String, mutable.HashMap[String, String]]): mutable.HashMap[String, Int]= {
    var SLOCMap =mutable.HashMap[String, Int]()
    var SLOCInfoMap =mutable.HashMap[String, Int]()
    println(fileInfo)

    var mediaSloc = 0
    var sumSloc = 0
    fileInfo.foreach{case (index, list) =>{
      SLOCMap += (index -> list("SLOC").toInt)
      sumSloc += list("SLOC").toInt
    }}
    val mapSortSmall = SLOCMap.toList.sortBy(_._2)
    println(mapSortSmall)
    val sum = fileInfo.size
    if(sum%2 == 0){
      val half = sum/2
      mediaSloc = (mapSortSmall.apply(half)._2+mapSortSmall.apply(half-1)._2)/2
    }else{
      val half = Math.ceil(sum/2).toInt
      mediaSloc = mapSortSmall.apply(half-1)._2
    }
    SLOCInfoMap += ("MediaSloc" -> mediaSloc)
    SLOCInfoMap += ("SumSloc" -> sumSloc)
    SLOCInfoMap
  }
  def getHeadAuthors(fileName:String):String = {
    var authors = ""
    val root = "E:\\graduate-design\\git-head\\"
    val path = root + fileName +".txt"
    val file = new File(path)
    val source = Source.fromFile(file)
    source.getLines.foreach(f => {
      authors += f
    })
    authors
  }
  //def generate(logs: )
  def generate(logs: Vector[String], path: String): Unit = {
    if (new File(path).delete) println("Delete existing result file.")
    val writer = new PrintWriter(new File(path))
    logs.foreach(s => writer.write(s + "\n"))
    writer.flush()
    writer.close()
  }

}
