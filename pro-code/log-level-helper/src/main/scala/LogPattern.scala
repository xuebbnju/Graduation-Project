
import scala.io.Source

/**
  * Author: xbb
  * Date: 18/12/18
  */
object LogPattern {
  val patterns: Vector[String] = {
    val source = Source.fromResource("pattern")
    val list = source.getLines.toVector
    source.close()
    list
  }
}
