package fr.toxic.spark.text.featurization

import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.DataFrame
import org.apache.lucene.analysis.Analyzer

/**
  * Created by mahjoubi on 12/06/18.
  */
class StopWordsRemoverTask(val inputColumn: String = "words",
                           val outputColumn: String = "words",
                           val stopWordsOption: String) {

  var stopWordsList: Array[String] = _

  def run(data: DataFrame): DataFrame = {
    stopWordsRemover(data, inputColumn, outputColumn, stopWordsList)
  }

  def stopWordsRemover(data: DataFrame, inputColumn: String, outputColumn: String, stopWords: Array[String]): DataFrame = {
    val removed = new StopWordsRemover().setInputCol(inputColumn).setOutputCol("stop_words_removed").transform(data)
    removed.drop(inputColumn).withColumnRenamed("stop_words_removed", outputColumn)
  }

  def defineStopWordsList(): StopWordsRemoverTask = {
    stopWordsList = if (stopWordsOption == "lucene") {
      Array("")
    } else {
      StopWordsRemover.loadDefaultStopWords("english")
    }
    this
  }


}
