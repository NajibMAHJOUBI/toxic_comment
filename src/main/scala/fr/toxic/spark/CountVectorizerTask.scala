package fr.toxic.spark

import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 12/06/18.
  */
class CountVectorizerTask(val inputColumn: String = "words", val outputColumn: String = "tf",
                          val minDF: Int = 2, val vocabSize: Int = 3) {

  def run(data: DataFrame): DataFrame = {
    countVectorizer(data, inputColumn, outputColumn, minDF, vocabSize)
  }

  def countVectorizer(data: DataFrame, inputColumn: String, outputColumn: String, minDF: Int, vocabSize: Int): DataFrame = {
    new CountVectorizer()
      .setInputCol(inputColumn)
      .setOutputCol(outputColumn)
      .setVocabSize(vocabSize)
      .setMinDF(minDF)
      .fit(data)
      .transform(data)
  }
}
