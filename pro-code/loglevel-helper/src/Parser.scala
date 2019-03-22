import java.io.{File, PrintWriter}
import java.util.concurrent.ConcurrentHashMap

import scala.collection.mutable
import scala.io.Source

/**
  * Author: xbb
  * Date: 18/12/18
  */
object Parser {
  val root = "E:\\graduate-design\\"
  val name = "git-test\\test2\\test3"

  def main(args: Array[String]): Unit = {
    val projectPath = root + name
    val outputPath = root + name + ".txt"
    val files = getJavaFiles(new File(projectPath)).toVector
    println(files.length)
    parseProject(files)
  }

  def getJavaFiles(file: File): Array[File] = {
    val files = file.listFiles
    files.filter(_.isFile).filter(_.toString.endsWith(".java")) ++ files.filter(_.isDirectory).flatMap(getJavaFiles)
  }

  def parseProject(files: Vector[File]): Unit = files.par.map(FileParser.parseFile).reduce(_ ++ _)
  //def generate(logs: )
}
