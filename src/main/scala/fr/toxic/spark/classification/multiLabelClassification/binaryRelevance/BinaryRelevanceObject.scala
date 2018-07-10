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

  def savePrediction(data: DataFrame, labels: Array[String], path: String): Unit = {
    val columnsToKeep: Set[Column] = (Set("id")
      ++ labels.map(name => s"label_$name").toSet
      ++ labels.map(name => s"prediction_$name").toSet
      )
      .map(name => col(name))

    data
      .select(columnsToKeep.toSeq: _*)
      .write.option("header", "true").mode("overwrite")
      .csv(s"$path")
  }

  def formatDataToMultiLabelMetrics(data: DataFrame, labels: Array[String]): RDD[(Array[Double], Array[Double])] = {
    val predictionColumns = labels.map(column => s"prediction_$column")
    val labelColumns = labels.map(column => s"label_$column")
    data.rdd.map(p => (predictionColumns.map(column => p.getDouble(p.fieldIndex(column))),
        labelColumns.map(column => p.getLong(p.fieldIndex(column)).toDouble)))
  }

  def multiLabelPrecision(data: DataFrame, labels: Array[String], criteria: String = "accuracy"): Unit= {
    val predictionAndLabels: RDD[(Array[Double], Array[Double])] = formatDataToMultiLabelMetrics(data, labels)

    val metrics = new MultilabelMetrics(predictionAndLabels)
    if (criteria == "recall"){
      println(s"Recall = ${metrics.recall}")
    } else if (criteria == "precision") {
      println(s"Precision = ${metrics.precision}")
    } else if (criteria == "f1Measure") {
      println(s"F1 measure = ${metrics.f1Measure}")
    } else {
      println(s"Accuracy = ${metrics.accuracy}")
    }
  }

}
