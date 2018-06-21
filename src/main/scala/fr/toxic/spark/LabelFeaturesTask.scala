package fr.toxic.spark

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.col

/**
  * Created by mahjoubi on 12/06/18.
  */
class LabelFeaturesTask(val oldLabelColumn: String, val newLabelColumn: String,
                        val oldFeatureColumn: String, val newFeatureColumn: String) {

  def run(data: DataFrame): DataFrame = {
    defineFeatures(defineLabel(data))
  }

  def defineLabel(data: DataFrame) = {
    data.withColumnRenamed(oldLabelColumn, newLabelColumn)
  }

  def defineFeatures(data: DataFrame) = {
    data.withColumnRenamed(oldFeatureColumn, newFeatureColumn)
  }

}
