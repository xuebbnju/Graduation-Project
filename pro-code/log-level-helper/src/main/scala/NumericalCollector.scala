
import com.github.javaparser.ast.expr.AssignExpr
import com.github.javaparser.ast.stmt.AssertStmt
import com.github.javaparser.ast.expr.MethodCallExpr
import com.github.javaparser.ast.expr.VariableDeclarationExpr
import com.github.javaparser.ast.visitor.VoidVisitorAdapter


import scala.collection.mutable
/**
  * Author: xbb
  * Date: 18/12/20
  */
class NumericalCollector extends VoidVisitorAdapter[mutable.HashMap[String, Int]] {
  override def visit(n: AssignExpr, arg: mutable.HashMap[String, Int]): Unit = {
    super.visit(n, arg)
        if(isJDBC(n)){
          if(arg("JDBCInBlock") == 0){
            arg.update("JDBCInBlock",1)
          }
          if(canUpdate(n.getBegin.get.line, arg.get("JDBCLine").get, arg.get("JDBCLine").get)) {
            arg.update("JDBCLine",n.getBegin.get.line)
          }
        }

  }
  override def visit(n: MethodCallExpr, arg: mutable.HashMap[String, Int]): Unit = {
    super.visit(n, arg)
    //    println(" MethodCallExpr:")
    //    println(n)

    if(n.getBegin.get.line != arg.get("LogLine").get){
        val exp = n.removeComment().asInstanceOf[MethodCallExpr]//get all expression of Ast
        val patterns = LogPattern.patterns.par.filter(_.r.findFirstIn(exp.toString).isDefined)//Match Log Statement
        if (patterns.nonEmpty){
          if(arg("LogInBlock") == 0) {
            arg.update("LogInBlock", 1)
          }
          arg.update("LogInBlockCount",arg.get("LogInBlockCount").get+1)
        }
      arg.update("MethodCallCount",arg.get("MethodCallCount").get+1)
      arg.update("MethodParameterCount",arg.get("MethodParameterCount").get+n.getArguments.toArray.length)
      if(isThread(n)){
        if(arg("ThreadInBlock") == 0){
          arg.update("JThreadInBlock",1)
        }
        if(canUpdate(n.getBegin.get.line, arg.get("ThreadLine").get, arg.get("LogLine").get)) {
          arg.update("ThreadLine",n.getBegin.get.line)
        }
      }
    }


  }
  override def visit(n: AssertStmt, arg: mutable.HashMap[String, Int]): Unit = {
    super.visit(n, arg)
    if(arg("AssertInBlock") == 0){
      arg.update("AssertInBlock",1)
    }
    if(canUpdate(n.getBegin.get.line, arg.get("AssertLine").get, arg.get("LogLine").get))   {
      arg.update("AssertLine",n.getBegin.get.line)
    }

  }
  override def visit(n: VariableDeclarationExpr, arg: mutable.HashMap[String, Int]): Unit = {
    super.visit(n, arg)
    //    println(" VariableDeclarationExpr:")
    //    println(n)
    arg.update("VariableDeclarationCount",arg.get("VariableDeclarationCount").get+1)
//    val exp = n.asInstanceOf[AssignExpr]
//    if(isJDBC(exp)){
//      if(canUpdate(n.getBegin.get.line, arg.get("JDBCLine").get, arg.get("JDBCLine").get)) {
//        arg.update("JDBCLine",n.getBegin.get.line)
//      }
//    }
  }
  def isThread (n: MethodCallExpr): Boolean = {
    val methodName = n.getName
    var res = false
    if (methodName != null && (methodName.equals("start") || methodName.equals("join"))) {
      if(n.getScope.getClass.toString.contains("java.lang.Thread")){
         res = true
      }
    }
    res
  }
  def canUpdate(newLine: Int, oldLine: Int, logLine:Int): Boolean = {
    var res = false
    val diff = logLine - newLine
    if(newLine > oldLine && newLine < logLine && diff <5){
      res = true
    }
    res
  }
  def isJDBC (n: AssignExpr): Boolean = {
    var res = false
    val lConn = n.getTarget.toString
    if (n.getOperator.equals("=") && lConn.contains(".")) {
      val methodCall = n.getTarget.asMethodCallExpr()
      val methodName = methodCall.getName
      if (methodName != null && methodName.equals("getConnection")) {
        if(methodCall.getScope.getClass.toString.contains("java.sql.DriverManager")){
          res = true
        }
      }
    }
    res
  }

}