package fr.toxic.spark.utils

import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 12/06/18.
  */
class LabelFeaturesTask(val oldLabelColumn: String, val newLabelColumn: String,
                        val oldFeatureColumn: String, val newFeatureColumn: String) {

  def run(data: DataFrame): DataFrame = {
    defineFeatures(defineLabel(data))
  }

  def defineLabel(data: DataFrame)= {
    data.withColumnRenamed(oldLabelColumn, newLabelColumn)
  }

  def defineFeatures(data: DataFrame) = {
    data.withColumnRenamed(oldFeatureColumn, newFeatureColumn)
  }

}
