package fr.toxic.spark.classification.stackingMethod

import fr.toxic.spark.utils.LoadDataSetTask
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

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

  def mergeData(spark: SparkSession, label: String): StackingMethodTask = {
    data = loadDataLabel(spark, label)
    classificationMethods.foreach(method => data = data.join(loadDataPredictionByLabel(spark, method, label), Seq("id")))
    data = data.drop("id")
    this
  }

  def loadDataPredictionByLabel(spark: SparkSession, method: String, label: String): DataFrame = {
    new LoadDataSetTask(s"$pathPrediction/$method", format = "csv")
      .run(spark, "prediction")
      .select(col("id"), col(s"prediction_$label").alias(s"prediction_$method"))
  }

  def loadDataLabel(spark: SparkSession, label: String): DataFrame = {
    new LoadDataSetTask(sourcePath = pathLabel).run(spark, "train")
      .select(col("id"), col(label).alias("label"))
  }

  def createDfLabelFeatures(spark: SparkSession, label: String): DataFrame = {
    val classificationMethodsBroadcast = spark.sparkContext.broadcast(classificationMethods)
    val features = (p: Row) => {
      StackingMethodObject.extractVector(p, classificationMethodsBroadcast.value)
    }
    val rdd = data.rdd.map(p => (p.getLong(p.fieldIndex("label")), features(p)))
    spark.createDataFrame(rdd).toDF(label, "features")
  }

}
