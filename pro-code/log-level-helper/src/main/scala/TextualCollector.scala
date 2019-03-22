
import com.github.javaparser.ast.stmt.ExpressionStmt
import com.github.javaparser.ast.stmt.ReturnStmt
import com.github.javaparser.ast.stmt.ThrowStmt
import com.github.javaparser.ast.stmt.AssertStmt
import com.github.javaparser.ast.expr.MethodCallExpr
import com.github.javaparser.ast.expr.VariableDeclarationExpr
import com.github.javaparser.ast.visitor.VoidVisitorAdapter
import com.github.javaparser.ast.body.Parameter


import scala.collection.mutable
/**
  * Author: xbb
  * Date: 18/12/20
  */
class TextualCollector extends VoidVisitorAdapter[mutable.HashMap[String, String]] {
  override def visit(n: VariableDeclarationExpr, arg: mutable.HashMap[String, String]): Unit = {
    super.visit(n, arg)
    val exp = n.removeComment().asInstanceOf[VariableDeclarationExpr]
    arg.update("VariableDeclarationName", arg("VariableDeclarationName")+exp.getVariables.get(0).getName+" ")
    arg.update("VariableDeclarationType", arg("VariableDeclarationType")+exp.getElementType.removeComment().toString()+" ")

  }
  override def visit(n: MethodCallExpr, arg: mutable.HashMap[String, String]): Unit = {
    super.visit(n, arg)
    if(n.getBegin.get.line != arg.get("logLine").get.toInt){
      val exp = n.removeComment().asInstanceOf[MethodCallExpr]
//      println("Method:")
//      println(n)
      var callerName = ""
      if (exp.getScope.isPresent()){
        if(exp.getScope.get.isMethodCallExpr){
          callerName = exp.getScope.get.asMethodCallExpr().getNameAsString
        }else if(exp.getScope.get.isObjectCreationExpr){
          callerName = exp.getScope.get.asObjectCreationExpr().getType.asString()
        }else{
          callerName = exp.getScope.get.removeComment().toString
        }
      }
//      if (exp.getScope.isPresent()){
//        if(exp.getScope.get.isMethodCallExpr){
//          callerName = exp.getScope.get.asMethodCallExpr().getNameAsString
//        }else if(exp.getScope.get.isObjectCreationExpr){
//          callerName = exp.getScope.get.asObjectCreationExpr().getType.asString()
//        }else{
//          callerName = exp.getScope.get().toString
//        }
//      }
      if(callerName.contains("we need to")){
        println(n)
        println(exp.getScope.get.removeComment())
      }
//      println(callerName)
//      println(".......")
      arg.update("MethodCallerName", arg("MethodCallerName")+callerName +" ")
      arg.update("MethodCallName", arg("MethodCallName")+n.getName+" ")
//      var str = ""
//      val argu = n.getArguments
//      for(i <- 0 until argu.size()) {
//        //println(argu.get(i).getMetaModel)
//        str += argu.get(i).getMetaModel+""
////        if (argu.get(i).isObjectCreationExpr ) {
////
////          str += argu.get(i).asObjectCreationExpr().getType +" "
////        }else if(argu.get(i).isClassExpr){
////          str += argu.get(i).asClassExpr().getType +" "
////        }
////        else if(argu.get(i).isMethodReferenceExpr){
////
////
////          str += argu.get(i).toString+" "
////        }else if(argu.get(i).isMethodCallExpr){
////
////          str += argu.get(i).toString+" "
////        }else{
////
////          if(argu.get(i).toString.contains("->")){
////            str += argu.get(i).toString.split("->",2)(0)+" "
////          }else {
////            str += argu.get(i).toString+" "
////          }
////
////        }
//
//
//      }
//
//      arg.update("MethodParameterName", arg("MethodParameterName")+str)
    }


  }
}
//class ParameterCollector extends VoidVisitorAdapter[mutable.HashMap[String, String]] {
//  override def visit(n: Parameter, arg: mutable.HashMap[String, String]): Unit = {
//    super.visit(n, arg)
//    println(123)
//    println(n)
//    println(n.getNameAsString)
//    println(n.getTypeAsString)
//  }
//}