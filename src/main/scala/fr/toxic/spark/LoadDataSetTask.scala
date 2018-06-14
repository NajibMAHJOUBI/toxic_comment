package fr.toxic.spark

import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by mahjoubi on 11/06/18.
  */
class LoadDataSetTask(val sourcePath: String) {

  def run(spark: SparkSession, dataset: String): DataFrame = {
    spark.read.parquet(s"${sourcePath}/${dataset}")
  }

}
