package fr.toxic.spark.classification.binaryRelevance

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

  def multiLabelPrecision(data: DataFrame, labels: Array[String]): Unit= {
    val predictionColumns = labels.map(column => s"prediction_$column")
    val labelColumns = labels.map(column => s"label_$column")
    val predictionAndLabels: RDD[(Array[Double], Array[Double])] =
      data.rdd.map(p => (predictionColumns.map(column => p.getDouble(p.fieldIndex(column))),
        labelColumns.map(column => p.getLong(p.fieldIndex(column)).toDouble)))

    val metrics = new MultilabelMetrics(predictionAndLabels)
    println(s"Accuracy = ${metrics.accuracy}")
    //    println(s"Recall = ${metrics.recall}")
    //    println(s"Precision = ${metrics.precision}")
    //    println(s"F1 measure = ${metrics.f1Measure}")
  }

}
