package fr.toxic.spark.classification.stackingMethod

import fr.toxic.spark.utils.LoadDataSetTask
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.linalg.{Vector, Vectors}

class StackingMethodLogisticRegressionTask(labels: Array[String], classificationMethods: Array[String], pathLabel: String, pathPrediction: String) {

  def run(spark: SparkSession): StackingMethodLogisticRegressionTask = {
    labels.foreach(label => {
      val data = mergeData(spark, label)





    })
    this
  }

  def loadDataPredictionByLabel(spark: SparkSession, method: String, label: String): DataFrame = {
    new LoadDataSetTask(s"$pathPrediction/$method", format= "csv")
      .run(spark, "prediction")
      .select(col("id"), col(s"prediction_$label"))
  }

  def loadDataLabel(spark: SparkSession, label: String): DataFrame = {
    new LoadDataSetTask(sourcePath = pathLabel).run(spark, "train")
      .select(col("id"), col(label).alias("label"))
  }

  def mergeData(spark: SparkSession, label: String): DataFrame = {
    var data: DataFrame = loadDataLabel(spark, label)
    classificationMethods.foreach(method => data = data.join(loadDataPredictionByLabel(spark, method, label), Seq("id")))
    data.drop("id")
  }

  def createLabelFeatures(dataFrame: DataFrame): Vector = {
    val u = Vectors.dense(Array(1.0, 2.0, 3.3))
    print(u)
    u
  }



}
