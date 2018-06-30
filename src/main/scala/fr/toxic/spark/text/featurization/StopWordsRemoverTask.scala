package fr.toxic.spark.text.featurization

import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 12/06/18.
  */
class StopWordsRemoverTask(val inputColumn: String = "words", val outputColumn: String = "words",
                           val stopWords: Array[String] = StopWordsRemover.loadDefaultStopWords("english")) {

  def run(data: DataFrame): DataFrame = {
    stopWordsRemover(data, inputColumn, outputColumn, stopWords)
  }

  def stopWordsRemover(data: DataFrame, inputColumn: String, outputColumn: String, stopWords: Array[String]): DataFrame = {
    val removed = new StopWordsRemover().setInputCol(inputColumn).setOutputCol("stop_words_removed").transform(data)
    removed.drop(inputColumn).withColumnRenamed("stop_words_removed", outputColumn)
  }

}
