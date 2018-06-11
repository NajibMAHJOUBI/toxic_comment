package fr.toxic.spark

import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by mahjoubi on 11/06/18.
  */
class LoadDataSet(val sourcePath: String = "/home/mahjoubi/Documents/github/toxic_comment/data/parquet") {

  def run(spark: SparkSession, dataset: String): DataFrame = {
    spark.read.parquet(s"${sourcePath}/${dataset}")
  }

}
