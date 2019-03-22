
import com.github.javaparser.ast.stmt.ReturnStmt
import com.github.javaparser.ast.stmt.ThrowStmt
import com.github.javaparser.ast.expr.{AssignExpr, VariableDeclarationExpr}
import com.github.javaparser.ast.visitor.VoidVisitorAdapter

import scala.collection.mutable
/**
  * Author: xbb
  * Date: 18/12/20
  */
class BooleanCollector  extends VoidVisitorAdapter[mutable.HashMap[String, Boolean]] {
  override def visit(n: AssignExpr, arg: mutable.HashMap[String, Boolean]): Unit = {
    super.visit(n, arg)
    //    println(" ExpressionStmt:")
    if(n.getOperator.equals("=") && !arg.get("SettingFlag").get){
      if(n.getTarget.equals(-1) || n.getTarget.equals(null) ||  n.getTarget.equals(false)){
        arg.update("SettingFlag",true)
      }
    }
  }
  override def visit(n: VariableDeclarationExpr, arg: mutable.HashMap[String, Boolean]): Unit = {
    super.visit(n, arg)
    //    println(" VariableDeclarationExpr:")

    if(!arg.get("SettingFlag").get){
       val exp = n.toString
      if(exp.contains("-1")||exp.contains("null")||exp.contains("fasle")){
        arg.update("SettingFlag",true)
      }
    }
  }
  override def visit(n: ReturnStmt, arg: mutable.HashMap[String, Boolean]): Unit = {
    super.visit(n, arg)
    //    println(" ReturnStmt:")
    //    println(n)
    if(!arg.get("ReturnInBlock").get){
      arg.update("ReturnInBlock", true)
    }
  }
  override def visit(n: ThrowStmt, arg: mutable.HashMap[String, Boolean]): Unit = {
    super.visit(n, arg)
    //    println(" ThrowStmt:")
    //    println(n)
    if(!arg.get("ThrowInBlock").get){
      arg.update("ThrowInBlock", true)
    }
  }

}