
import java.io.File

import FileParser.regex

import scala.collection.mutable
import scala.io.Source

/**
  * Author: xbb
  * Date: 18/12/18
  */



object FileParser {
  private val regex = "^[\\^\\w]+\\s*(.*)\\s*\\((.*)\\s+\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2} .\\d{4}\\s+(\\d+)\\) (.*)$".r

  def parseFile(file: File,fileInfo: mutable.LinkedHashMap[String, mutable.HashMap[String, String]]): mutable.LinkedHashMap[String, mutable.HashMap[String, String]] = {
//   println("............fileStart.......")



    //println(file.getAbsolutePath)
    val nameMap: mutable.HashMap[Int, String] = mutable.HashMap()
    var fileAuthorSet = mutable.Set[String]()
    var fileInfoMap = mutable.LinkedHashMap[String, mutable.HashMap[String, String]]()
//    val parentPath = file.getParent
//    val len = file.getName.size
//    val blamePath = parentPath + "\\" + file.getName.substring(0,len-5) + ".blame"
//    //println(blamePath)
//    val sourceBlame = Source.fromFile(new File(blamePath))

//    sourceBlame.close()
    val source = Source.fromFile(file)
    var code: String = ""
    var fileLines = 0
    source.getLines.foreach(f => regex.findAllIn(f).matchData.foreach(p => {
      nameMap += (p.group(3).toInt -> p.group(2).trim)// order and author
      if(p.group(4) == "package ${package};"){
        code +=   "\n"
      }else{
        code += p.group(4) + "\n"//code contnent
      }

      fileLines = fileLines+1
    }))
//    source.getLines.foreach(f => {

//    })
    source.close()
//    val infoMap = mutable.LinkedHashMap[String, mutable.HashMap[String, String]]()

    val infoMap = ASTParser.parseAST(code)


    if (infoMap.size != 0) {
      addFilesMetrics(infoMap, fileLines, file.getAbsolutePath,nameMap,fileAuthorSet,fileInfoMap)
      val fileMap: mutable.HashMap[String, String] = mutable.HashMap()
      val logdensity = ((infoMap.size.toDouble) / fileLines.toDouble).formatted("%.3f")
      fileMap += ("Logdensity" -> logdensity)
      fileMap += ("SLOC" -> fileLines.toString)
      fileMap += ("Authors" -> fileAuthorSet.mkString(";"))
      //println(fileMap("Authors"))
      fileInfo += (file.getAbsolutePath -> fileMap)
    }
//    }else{
//      println(fileInfoMap)
//      println("..............fileEnd.......")
//      return mutable.LinkedHashMap[String, mutable.HashMap[String, String]]()
//    }
//
//      println(fileInfo)
//      println("..............fileEnd.......")
      return  fileInfoMap

  }
  def addFilesMetrics (infoMap: mutable.LinkedHashMap[Int, mutable.HashMap[String, String]], fileLines: Int, fileName: String,nameMap: mutable.HashMap[Int, String],fileAuthorSet: mutable.Set[String],fileInfoMap:mutable.LinkedHashMap[String, mutable.HashMap[String, String]]): Unit = {
    val logNumber = infoMap.size-1
    val logLengthSum = getSum(infoMap,"LogLength")
    val logParameterSum = getSum(infoMap,"LogParameterCount")
    var blockLOC = 0
    var averageLogLength = 0
    var logdensity = "0"
    var averageLogParameterCount = 0
    var levleMap =mutable.HashMap[String, Int]()

    for(c <- infoMap){
      c._2 += ("LogNumber" -> logNumber.toString)
      if(logNumber > 0){
        averageLogLength = (logLengthSum - c._2.get("LogLength").get.toInt)/logNumber
        averageLogParameterCount = (logParameterSum - c._2.get("LogParameterCount").get.toInt)/logNumber
        logdensity = ((c._2.get("BlockSLOC").get.toDouble)/fileLines.toDouble).formatted("%.2f")
      }
      val logLevle = c._2.get("LogLevel").get
      if(levleMap.contains(logLevle)) {
        levleMap += (logLevle -> (levleMap(logLevle) + 1))
      } else{
        levleMap += (logLevle -> 1)
      }
      c._2 += ("Logdensity" -> logdensity)
      c._2 += ("AverageLogLength" -> averageLogLength.toString)
      c._2 += ("LogNumber" -> logNumber.toString)
      c._2 += ("AverageeLogParameterCount" -> averageLogParameterCount.toString)
    }
    val averageMap = getAverageLevel(levleMap, infoMap.size)
    for(c<- infoMap){
      val logLevle = c._2.get("LogLevel").get
      if(logNumber > 0) {
        c._2 += ("MaxLogLevel" -> averageMap(logLevle))
      }else{
        c._2 += ("MaxLogLevel" -> " ")
      }
      c._2 += ("Author" -> nameMap(c._1))
      c._2 += ("FileName" -> fileName)
      fileAuthorSet += (nameMap(c._1))
      val key = c._1.toString + " "+ c._2("Content")
      fileInfoMap += (key-> c._2)
    }

  }
  def getAverageLevel (levleMap: mutable.HashMap[String, Int],total:Int): mutable.HashMap[String, String] ={
    val keys=levleMap.map(_._1)
    var averageMap =mutable.HashMap[String, String]()
    keys.foreach(key => {
      levleMap += (key -> (levleMap(key) - 1))
      val mapSortBig = levleMap.toList.sortBy(-_._2)
      averageMap += (key -> mapSortBig.head._1)
      levleMap += (key -> (levleMap(key) + 1))
    })

    averageMap
  }
//  def getMediaLevel (levleMap: mutable.HashMap[String, Int],total:Int): String = {
//    val mapSortSmall = levleMap.toList.sortBy(_._2)
//    var sum = 0
//    val half = total/2
//    var res = ""
//    var flag = true;
//    for(c <- mapSortSmall){
//      sum += c._2
//      if(flag && sum>half){
//        flag = false
//        res = c._1
//      }
//    }
//    res
//  }
  def getSum(infoMap: mutable.LinkedHashMap[Int, mutable.HashMap[String, String]],key: String ): Int = {
    var res = 0
    for(c <- infoMap){
      res += c._2.get(key).get.toInt
    }
    res
  }
}
