package fr.toxic.spark.classification.classifierChains

import fr.toxic.spark.classification.task.LogisticRegressionTask
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

//class ClassifierChainsTask(val addedLabelColumns: String,
//                           val outputColumn: String,
//                           val featureColumn: String,
//                           val newFeaturesColumn: String) {
//
//  def run(data: DataFrame): DataFrame = {
//    modifyFeatures(data)
//  }
//
//  def modifyFeatures(data: DataFrame): DataFrame = {
//    val udfNewFeatures = udf((vector: Vector, value: Long) => ClassifierChainsObject.createNewFeatures(vector, value))
//    data.withColumn(newFeaturesColumn, udfNewFeatures(col(featureColumn), col(addedLabelColumns)))
//  }
//
//
//}

class ClassifierChainsLogisticRegressionTask(val labelColumns: Array[String], val featureColumn: String, val methodValidation: String = "simple", val savePath: String) {

  def run(data: DataFrame): Unit = {
    labelColumns.map(label => {
      val dataSet = createNewDataSet(data, label: String)
      val logisticRegression = new LogisticRegressionTask(labelColumn = s"label_$label", featureColumn=featureColumn,
        predictionColumn = s"prediction_$label")
      logisticRegression.defineModel()
      logisticRegression.fit(data)
      logisticRegression.saveModel(s"$savePath/$label")
    })
  }

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