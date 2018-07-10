package fr.toxic.spark.classification.multiLabelClassification.binaryRelevance

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.MultilabelMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions.col


object BinaryRelevanceObject {

  def getPredictionProbability(probability: Vector, prediction: Double): Double = {
    probability(prediction.toInt)
  }

  def createLabel(data: DataFrame, column: String): DataFrame = {
    data.withColumnRenamed(column, s"label_$column")
  }

}
