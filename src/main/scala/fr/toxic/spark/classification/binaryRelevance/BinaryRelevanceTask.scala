package fr.toxic.spark.classification.binaryRelevance

import org.apache.spark.sql.DataFrame

class BinaryRelevanceTask(val columns: Array[String],
                          val savePath: String,
                          val featureColumn: String,
                          val methodValidation: String) {

  var prediction: DataFrame = _

  def getPrediction: DataFrame = {
    prediction
  }

  def getColumns: Array[String] = {
    columns
  }

  def getFeatureColumn: String = {
    featureColumn
  }

  def getMethodValidation: String = {
    methodValidation
  }
}
