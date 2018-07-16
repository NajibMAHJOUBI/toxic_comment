package fr.toxic.spark.text.featurization

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame

import scala.math.log

class TextFeaturizationTask {

  var prediction: DataFrame = _

  def getPrediction: DataFrame = {
    prediction
  }

  def logVector(u: Vector): Vector = {
    val size = u.size
    val indices = u.toSparse.indices
    val values = u.toSparse.values.map((x: Double) => log(x))
    Vectors.sparse(size, indices, values)
  }

}
