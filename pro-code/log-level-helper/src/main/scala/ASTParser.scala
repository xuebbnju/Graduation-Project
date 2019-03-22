import com.github.javaparser.JavaParser
import com.github.javaparser.ast.CompilationUnit
import com.github.javaparser.ast.stmt._
import com.github.javaparser.ast.visitor.VoidVisitorAdapter
import com.github.javaparser.ast.PackageDeclaration
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration
import com.github.javaparser.ast.Node
import com.github.javaparser.ast.expr.{MethodCallExpr, NameExpr}
import com.github.javaparser.ast.stmt.{ForeachStmt,ForStmt,WhileStmt}
import scala.collection.mutable

/**
  * Author: xbb
  * Date: 18/12/18
  */

object ASTParser {
  def parseAST(code: String): mutable.LinkedHashMap[Int, mutable.HashMap[String, String]] = {
    try {
      JavaParser.parse(code)
      // Use and close file
    } catch {
      case ex: Exception => {

        println(code)}
    }
    var cu = JavaParser.parse(code)
    if(cu == null) return null
    val infoMap: mutable.LinkedHashMap[Int, mutable.HashMap[String, String]] = mutable.LinkedHashMap()
    val map:  mutable.HashMap[String, String] = new mutable.HashMap[String, String]()
    cu.accept(new LogCollector, infoMap)//visit Ast
    map += ("PackageName" -> " ")
    map += ("ClassName" -> " ")
    cu.accept(new PackagePart, map)
    cu.accept(new ClassPart, map)
    infoMap.foreach { case (index, list) => {
      list += ("PackageName" -> map("PackageName"))
      list += ("ClassName" -> map("ClassName"))
    }}
    infoMap
  }

  class LogCollector extends VoidVisitorAdapter[mutable.LinkedHashMap[Int, mutable.HashMap[String, String]]] {
    val checkTypeLine = Array("AssertLine","ThreadLine","JDBCLine")
    override def visit(n: MethodCallExpr, arg: mutable.LinkedHashMap[Int, mutable.HashMap[String, String]]): Unit = {
      super.visit(n, arg)
      val exp = n.removeComment().asInstanceOf[MethodCallExpr]//get all expression of Ast
      val patterns = LogPattern.patterns.par.filter(_.r.findFirstIn(exp.toString).isDefined)//Match Log Statement
            if (patterns.nonEmpty) {
              if (exp.toString.split("\n").length > 1) {
                exp.getChildNodes.forEach(f => f.accept(this, arg))
              } else {
                val textualMap: mutable.HashMap[String, String] = mutable.HashMap()
                val booleanMap: mutable.HashMap[String, Boolean] = mutable.HashMap()
                val numericalMap: mutable.HashMap[String, Int] = mutable.HashMap()
                initContextMap(exp,numericalMap, booleanMap,textualMap)
                getBlockType(exp,textualMap,numericalMap,booleanMap)
                patterns.head.r.findFirstMatchIn(exp.toString).foreach(p=>{

                  val logLevel = p.group(1).toLowerCase match {
                    case "error"|"severe" => "error"
                    case "finest"|"trace" => "trace"
                    case "fine"|"finer"|"debug" => "debug"
                    case "info"|"config" => "info"
                    case "warn" => "warn"
                    case "fatal"=> "fatal"
                  }

                  //println("运行1")
                  textualMap += ("LogLevel" -> logLevel)
                })
                //println("运行2")
                for(c <- booleanMap){
                  textualMap += (c._1 -> (if (c._2) "1" else "0"))
                }
                for(c <- numericalMap){
                  textualMap += (c._1 -> c._2.toString)
                }
                textualMap += ("Content" -> exp.toString)
                textualMap += ("LogArg" -> exp.getArguments().toString)
                arg += (exp.getBegin.get.line -> textualMap)

              }
            }
    }
    def initContextMap (exp: MethodCallExpr ,numericalMap: mutable.HashMap[String, Int],booleanMap: mutable.HashMap[String, Boolean],textualMap: mutable.HashMap[String, String]): Unit ={
      booleanMap += ("ReturnInBlock" -> false)
      booleanMap += ("ThrowInBlock" -> false)
      booleanMap += ("SettingFlag" -> false)

      for(c<-checkTypeLine){
        numericalMap += (c -> -1)
      }
      numericalMap += ("AssertInBlock" -> 0)
      numericalMap += ("ThreadInBlock" -> 0)
      numericalMap += ("JDBCInBlock" -> 0)
      numericalMap += ("LogInBlock" -> 0)
      numericalMap += ("LogInBlockCount" -> 0)
      numericalMap += ("MethodParameterCount" -> 0)
      numericalMap += ("VariableDeclarationCount" -> 0)
      numericalMap += ("MethodCallCount" -> 0)
      numericalMap += ("LogLine" -> exp.getBegin.get.line)
      numericalMap += ("LogParameterCount" -> exp.getArguments.toArray.length)
      numericalMap += ("LogLength" -> exp.toString.length)
      textualMap += ("MethodCallName" -> "")
      textualMap += ("MethodCallerName" -> "")
//      textualMap += ("MethodParameterName" -> " ")
      textualMap += ("VariableDeclarationType" -> "")
      textualMap += ("VariableDeclarationName" -> "")
      textualMap += ("LoopCondition" -> "")
      textualMap += ("logLine" -> exp.getBegin.get.line.toString)
    }
    def getBlockType (exp: com.github.javaparser.ast.Node,textualMap: mutable.HashMap[String, String],numericalMap: mutable.HashMap[String, Int],booleanMap:mutable.HashMap[String, Boolean]):  String = {
      val parentNode = exp.getParentNode.get
      val typeName = parentNode.getMetaModel.getTypeName.toLowerCase
      val classReg = "class|field|constructor|initializer|compilationunit".r
      var blockName = ""
      val  deleReg = "\\>|\\<|\\.|\\=|\\&&|\\(|\\)|!|\\|"
      typeName match {
        case typeName if(typeName.contains("for")) => {
          blockName ="ForBlock"
          //println(typeName)
          var loopCondition = ""
          if(typeName == "foreachstmt"){
            loopCondition = parentNode.asInstanceOf[ForeachStmt].getIterable().toString
          }else if(typeName == "forstmt"){
            loopCondition = parentNode.asInstanceOf[ForStmt].	getCompare().get().toString
          }
          //println(loopCondition)
          loopCondition = loopCondition.replaceAll(deleReg, ";").replaceAll(";", " ")
          //(loopCondition)
          textualMap.update("LoopCondition",loopCondition)
        }
        case typeName if(typeName.contains("switch")) => {
          blockName ="SwitchBlock"
        }
        case typeName if(typeName.contains("catch")) => blockName ="CatchBlock"
        case typeName if(typeName.contains("try")) => blockName ="TryBlock"
        case typeName if(typeName.contains("do")) => {
          blockName ="DoBlock"
        }
        case typeName if(typeName.contains("method")||typeName.contains("objectcreation")) => {
          blockName ="MethodBlock"
        }
        case typeName if(typeName.contains("if")) => {
          blockName ="IfBlock"
        }
        case typeName if(typeName.contains("while")) => {
          blockName ="WhileBlock"
          println(typeName)
          var loopCondition = ""
          loopCondition = parentNode.asInstanceOf[WhileStmt].getCondition.toString
          println(loopCondition)
          loopCondition = loopCondition.replaceAll(deleReg, ";").replaceAll(";", " ")
          println(loopCondition)
          textualMap.update("LoopCondition",loopCondition)

        }
        case typeName if(typeName.contains("synchronized")) => blockName ="SynchronizedBlock"
        case typeName if(classReg.pattern.matcher(typeName).matches()) => blockName ="ClassBlock"
        case _ => ""
      }
      if(blockName == ""){
        blockName = getBlockType(parentNode,textualMap,numericalMap,booleanMap)
      }else{
        parentNode.accept(new BooleanCollector,booleanMap)
        parentNode.accept(new NumericalCollector,numericalMap)
        parentNode.accept(new TextualCollector,textualMap)
        var exceptiontype= ""
        textualMap += ("BlockType" -> blockName)
        if(typeName.equals("catchclause")){
          val n = parentNode.asInstanceOf[CatchClause]
          exceptiontype = n.getParameter.getChildNodes.get(0).toString()
        }
        textualMap += ("ExceptionType" -> exceptiontype)
        val checkType = getCheckType(blockName, numericalMap)
        textualMap += ("CheckType" -> checkType)
        numericalMap += ("BlockSLOC" -> (parentNode.getEnd.get.line - parentNode.getBegin.get.line + 1))
      }

      return blockName
    }
    def getBlockCondition(exp: com.github.javaparser.ast.Node,typeN:String,caseN:Int): String = {

    }
    def canUpdate(newLine: Int, oldLine: Int, logLine:Int): Boolean = {
      var res = false
      val diff = logLine - newLine
      if(newLine > oldLine && newLine < logLine && diff <5){
        res = true
      }
      res
    }
    def getCheckType (name: String, numericalMap: mutable.HashMap[String, Int]): String ={
      var lines = -1
      var lineType = ""
      var res = ""
      for(c<-checkTypeLine){
        if(numericalMap.get(c).get>lines){
          lines = numericalMap.get(c).get
          lineType = c.substring(0,c.length-4)+"Check"
        }
      }
      if(lines != -1){
        res = lineType
      }else{
        res = name match{
          case "IfBlock"|"SwitchBlock" => "LogicBranch"
          case "TryBlock"|"CatchBlock" => "Exception"
          case "ForBlock"|"WhileBlock"|" DoBlock" => "LoopCheck"
          case "MethodBlock" => "CriticalMethodCheck"
          case "SynchronizedBlock" => "AssertCheck"
          case _ => "CriticalClassesCheck"
        }
      }
      res
    }
  }
  class ClassPart extends VoidVisitorAdapter[mutable.HashMap[String, String]]{
    override def visit(n: ClassOrInterfaceDeclaration ,arg: mutable.HashMap[String, String]): Unit = {
      super.visit(n,arg)
      arg += ("ClassName" -> n.getNameAsString())
    }
  }
  class PackagePart extends VoidVisitorAdapter[mutable.HashMap[String, String]]{
    override def visit(n: PackageDeclaration ,arg: mutable.HashMap[String, String]): Unit = {
      super.visit(n,arg)
      arg += ("PackageName" -> n.getNameAsString().replaceAll("\\."," "))
    }
  }
  class NamePart extends VoidVisitorAdapter[mutable.HashMap[String, String]]{
    override def visit(n: NameExpr ,arg: mutable.HashMap[String, String]): Unit = {
      super.visit(n,arg)
      println(n)
    }
  }
}
