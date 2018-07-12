package fr.toxic.spark.classification.stackingMethod

import org.apache.spark.sql.SparkSession

trait StackingMethodFactory {

  def run(spark: SparkSession): StackingMethodFactory

}
