import java.io.File

import scala.collection.mutable
import scala.io.Source

/**
  * Author: xbb
  * Date: 18/12/18
  */



object FileParser {

  def parseFile(file: File): Unit = {

    val source = Source.fromFile(file)
    var code: String = ""
    source.getLines.foreach(f => code += f + "\n")

    source.close()


  }
}
