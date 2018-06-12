package fr.toxic.spark

import org.apache.spark.ml.feature.IDF
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 12/06/18.
  */
class TfIdfTask(val inputColumn: String = "tf", val outputColumn: String = "tf_idf") {

  def run(data: DataFrame): DataFrame = {
    computeTfIdf(data, inputColumn, outputColumn)
  }

  def computeTfIdf(data: DataFrame, inputColumn: String, outputColumn: String): DataFrame = {
    new IDF().setInputCol(inputColumn).setOutputCol(outputColumn).fit(data).transform(data)
  }
}
