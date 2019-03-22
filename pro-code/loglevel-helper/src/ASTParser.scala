import com.github.javaparser.JavaParser
import com.github.javaparser.ast.stmt._
import com.github.javaparser.ast.visitor.VoidVisitorAdapter
import com.github.javaparser.ast.body.MethodDeclaration
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration

import scala.collection.mutable

/**
  * Author: xbb
  * Date: 18/12/18
  */

object ASTParser {
  def parseAST(code: String): Unit = {
    var cu = JavaParser.parse(code)
    if(cu == null) return null
    val infoMap: mutable.LinkedHashMap[Int, mutable.ArrayBuffer[String]] = mutable.LinkedHashMap()
    cu.accept(new LogCollector, infoMap)//visit Ast
    if (infoMap.isEmpty) return null
  }

  class LogCollector extends VoidVisitorAdapter[mutable.LinkedHashMap[Int, mutable.ArrayBuffer[String]]] {
    override def visit(n: ExpressionStmt, arg: mutable.LinkedHashMap[Int, mutable.ArrayBuffer[String]]): Unit = {
      super.visit(n, arg)
      val exp = n.removeComment().asInstanceOf[ExpressionStmt]//get all expression of Ast
      println(exp)
    }
  }

}
