package fr.toxic.spark.classification.stackingMethod

import org.apache.spark.sql.{DataFrame, SparkSession}

trait StackingMethodFactory {

  def run(spark: SparkSession): StackingMethodFactory

  def computeModel(data: DataFrame, label: String): StackingMethodFactory

}
