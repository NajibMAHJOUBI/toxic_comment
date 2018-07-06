package fr.toxic.spark.classification.binaryRelevance
l
import org.apache.spark.sql.DataFrame

class BinaryRelevanceTask {

  var prediction: DataFrame = _

  def getPrediction: DataFrame = {
    prediction
  }


}
