package fr.toxic.spark.classification.classifierChains

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

class ClassifierChainsTask(val labelColumns: Array[String],
                           val featureColumn: String,
                           val methodValidation: String,
                           val savePath: String) {

  def modifyFeatures(data: DataFrame, label: String): DataFrame = {
    val udfNewFeatures = udf((vector: Vector, value: Long) => ClassifierChainsObject.createNewFeatures(vector, value))
    val dataSet = data.withColumn("newFeatures", udfNewFeatures(col(featureColumn), col(label))).drop(featureColumn)
    dataSet.withColumnRenamed("newFeatures", featureColumn)
  }

  def createNewDataSet(data: DataFrame, label: String) : DataFrame = {
    val addLabelColumns = labelColumns.toBuffer - label
    var dataSet = data
    for (column <- addLabelColumns) {
      dataSet = modifyFeatures(dataSet, column)
    }
    dataSet
  }
}
