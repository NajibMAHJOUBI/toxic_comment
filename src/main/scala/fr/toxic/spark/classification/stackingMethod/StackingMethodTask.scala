package fr.toxic.spark.classification.stackingMethod

import fr.toxic.spark.utils.LoadDataSetTask
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col

class StackingMethodTask(val labels: Array[String],
                         val classificationMethods: Array[String],
                         val pathLabel: String,
                         val pathPrediction: String) {

  var data: DataFrame = _

  def getData: DataFrame = {
    data
  }

  def getLabels: Array[String] = {
    labels
  }

  def loadDataPredictionByLabel(spark: SparkSession, method: String, label: String): DataFrame = {
    new LoadDataSetTask(s"$pathPrediction/$method", format= "csv")
      .run(spark, "prediction")
      .select(col("id"), col(s"prediction_$label").alias(s"prediction_$method"))
  }

  def loadDataLabel(spark: SparkSession, label: String): DataFrame = {
    new LoadDataSetTask(sourcePath = pathLabel).run(spark, "train")
      .select(col("id"), col(label).alias("label"))
  }

  def mergeData(spark: SparkSession, label: String): StackingMethodTask = {
    data = loadDataLabel(spark, label)
    classificationMethods.foreach(method => data = data.join(loadDataPredictionByLabel(spark, method, label), Seq("id")))
    data = data.drop("id")
    this
  }

  def createLabelFeatures(dataFrame: DataFrame): Vector = {
    val u = Vectors.dense(Array(1.0, 2.0, 3.3))
    print(u)
    u
  }

}
