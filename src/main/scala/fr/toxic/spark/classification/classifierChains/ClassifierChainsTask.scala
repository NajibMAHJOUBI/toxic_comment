package fr.toxic.spark.classification.classifierChains

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

class ClassifierChainsTask(val addedLabelColumns: String,
                           val outputColumn: String,
                           val featureColumn: String,
                           val newFeaturesColumn: String) {

  def run(data: DataFrame): DataFrame = {
    modifyFeatures(data)
  }

  def modifyFeatures(data: DataFrame): DataFrame = {
    val udfNewFeatures = udf((vector: Vector, value: Long) => ClassifierChainsObject.createNewFeatures(vector, value))
    data.withColumn(newFeaturesColumn, udfNewFeatures(col(featureColumn), col(addedLabelColumns)))
  }


}
