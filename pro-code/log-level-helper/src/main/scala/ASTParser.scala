import com.github.javaparser.JavaParser
import com.github.javaparser.ast.CompilationUnit
import com.github.javaparser.ast.stmt._
import com.github.javaparser.ast.visitor.VoidVisitorAdapter
import com.github.javaparser.ast.PackageDeclaration
import com.github.javaparser.ast.body.{ClassOrInterfaceDeclaration,MethodDeclaration}
import com.github.javaparser.ast.Node
import com.github.javaparser.ast.expr.{MethodCallExpr, NameExpr,ObjectCreationExpr}
import com.github.javaparser.ast.stmt.{ForeachStmt,ForStmt,WhileStmt,DoStmt,SwitchEntryStmt,IfStmt}
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
      textualMap += ("LogicBranchCondition" -> "")
      textualMap += ("MthodBlockType" -> "")
      textualMap += ("ExceptionType" -> "")
      textualMap += ("logLine" -> exp.getBegin.get.line.toString)
    }
    def getBlockType (exp: com.github.javaparser.ast.Node,textualMap: mutable.HashMap[String, String],numericalMap: mutable.HashMap[String, Int],booleanMap:mutable.HashMap[String, Boolean]):  String = {
      val parentNode = exp.getParentNode.get.removeComment()
      val typeName = parentNode.getMetaModel.getTypeName.toLowerCase
      val classReg = "class|field|constructor|initializer|compilationunit".r
      var blockName = ""
      //println(typeName)
      typeName match {
        case typeName if(typeName.contains("for")) => {
          blockName ="ForBlock"
//          println(typeName)
//          println(parentNode)
          var loopCondition = ""
          if(typeName == "foreachstmt"){
            loopCondition = parentNode.asInstanceOf[ForeachStmt].getIterable().removeComment().toString
          }else if(typeName == "forstmt"){
            if(parentNode.asInstanceOf[ForStmt].getCompare.isPresent){
              loopCondition = parentNode.asInstanceOf[ForStmt].getCompare.toString
            }
          }
          //println(loopCondition)
          addCondition(loopCondition,textualMap,"LoopCondition")

        }
        case typeName if(typeName.contains("switch")) => {
          blockName ="SwitchBlock"
          //println(typeName)
          //println(parentNode)
          var loopCondition = parentNode.asInstanceOf[SwitchEntryStmt].getLabel.toString
          //println(loopCondition)
          addCondition(loopCondition,textualMap,"LogicBranchCondition")
        }
        case typeName if(typeName.contains("catch")) => {
          blockName ="CatchBlock"
          var exceptiontype =  parentNode.asInstanceOf[CatchClause].getParameter.getChildNodes.get(0).toString()
          addCondition(exceptiontype,textualMap,"ExceptionType")
        }
        case typeName if(typeName.contains("try")) => {
          blockName ="TryBlock"
//          var exceptiontype =  parentNode.asInstanceOf[TryStmt].getCatchClauses.asInstanceOf[CatchClause].getParameter.getChildNodes.get(0).toString()
//          addCondition(exceptiontype,textualMap,"ExceptionType")
        }
        case typeName if(typeName.contains("do")) => {
          blockName ="DoBlock"
          //println(typeName)
          var loopCondition = parentNode.asInstanceOf[DoStmt].getCondition.toString
          //println(loopCondition)
          addCondition(loopCondition,textualMap,"LoopCondition")
        }
        case typeName if(typeName.contains("method")||typeName.contains("objectcreation")) => {
          blockName ="MethodBlock"
          //println(typeName)
          //println(parentNode)
          var loopCondition = ""
          if(typeName == "methoddeclaration"){
            loopCondition = parentNode.asInstanceOf[MethodDeclaration].getType().removeComment().toString
          }else if(typeName == "objectcreation"){
            loopCondition = parentNode.asInstanceOf[ObjectCreationExpr].getType().removeComment().toString
          }
          //println(loopCondition)
          addCondition(loopCondition,textualMap,"MthodBlockType")
        }
        case typeName if(typeName.contains("if")) => {
          blockName ="IfBlock"
         // println(typeName)
          //println(parentNode)

//          val str:  mutable.HashMap[String, String] = new mutable.HashMap[String, String]()
//          str += ("Content" -> "")
//          parentNode.asInstanceOf[IfStmt].getCondition.removeComment().accept(new ExpressPart,str)
//          println(str)
          var loopCondition = parentNode.asInstanceOf[IfStmt].getCondition.removeComment().removeComment().toString
          //println(loopCondition)
          addCondition(loopCondition,textualMap,"LogicBranchCondition")
          //println(textualMap("LogicBranchCondition"))
        }
        case typeName if(typeName.contains("while")) => {
          blockName ="WhileBlock"
          //println(typeName)
          var loopCondition = parentNode.asInstanceOf[WhileStmt].getCondition.toString
          //println(loopCondition)
          addCondition(loopCondition,textualMap,"LoopCondition")

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
        textualMap += ("BlockType" -> blockName)
        val checkType = getCheckType(blockName, numericalMap)
        textualMap += ("CheckType" -> checkType)
        numericalMap += ("BlockSLOC" -> (parentNode.getEnd.get.line - parentNode.getBegin.get.line + 1))
      }

      return blockName
    }
    def addCondition(inputS:String,textualMap: mutable.HashMap[String, String],keyName:String): Unit = {
      var loopCondition = ""
      val  deleReg = "\\p{P}|\\r|\\n|\\>|\\<|\\.|\\=|\\&|\\(|\\)|!|\\|\\//"
      loopCondition = inputS.replaceAll(deleReg, ";").replaceAll(";", " ")
//      if(keyName =="LogicBranchCondition" || keyName =="MthodBlockType"){
//
//      }
      textualMap.update(keyName,loopCondition)
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
  class ExpressPart extends VoidVisitorAdapter[mutable.HashMap[String, String]]{
    override def visit(n: ExpressionStmt ,arg: mutable.HashMap[String, String]): Unit = {
      super.visit(n,arg)
      println(n)
      arg.update("Content", arg("Content")+" "+n.toString())
    }
  }
}
