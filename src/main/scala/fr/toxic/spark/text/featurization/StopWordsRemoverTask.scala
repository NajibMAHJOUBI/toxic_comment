package fr.toxic.spark.text.featurization

import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.DataFrame
import org.apache.lucene.analysis.en.EnglishAnalyzer

/**
  * Created by mahjoubi on 12/06/18.
  *
  * Stop words remover task
  *
  */


class StopWordsRemoverTask(val inputColumn: String = "words", val outputColumn: String = "words",
                           val stopWordsOption: String) {

  var stopWords: Array[String] = _

  def run(data: DataFrame): DataFrame = {
    defineStopWordsList()
    stopWordsRemover(data)
  }

  def stopWordsRemover(data: DataFrame): DataFrame = {
    val removed = new StopWordsRemover()
      .setInputCol(inputColumn)
      .setOutputCol("stop_words_removed")
      .setStopWords(stopWords)

     removed.transform(data).drop(inputColumn).withColumnRenamed("stop_words_removed", outputColumn)
  }

  def defineStopWordsList(): StopWordsRemoverTask = {
    stopWords = if (stopWordsOption == "lucene") {
      new EnglishAnalyzer().getStopwordSet().toString.split(",")
    } else {
      StopWordsRemover.loadDefaultStopWords("english")
    }
    this
  }

  def getStopWords: Array[String] = {
    stopWords
  }

}
