package fr.toxic.spark.utils

import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by mahjoubi on 11/06/18.
  */
class LoadDataSetTask(val sourcePath: String, val format: String = "parquet") {

  private var data: DataFrame = _

  def run(spark: SparkSession, dataset: String): DataFrame = {
    if (format =="csv") {
      data = spark.read.option("header", "true").csv(s"${sourcePath}/${dataset}")
    } else {
      data = spark.read.parquet(s"${sourcePath}/${dataset}")
    }
    data
  }

}
