package fr.toxic.spark

import org.apache.spark
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by mahjoubi on 12/06/18.
  */
class CountVectorizerTask(val inputColumn: String = "words", val outputColumn: String = "tf",
                          val minDF: Int = 2, val vocabSize: Int = 3) {

  private var countVectorizer: CountVectorizer = _
  private var countVectorizerModel: CountVectorizerModel = _
  private var transform: DataFrame = _

  def run(data: DataFrame): Unit = {
    countVectorizer(data, inputColumn, outputColumn, minDF, vocabSize)
    defineModel()
    fit(data)
    transform(data)
  }

  def defineModel(): CountVectorizerTask = {
    countVectorizer =  new CountVectorizer()
      .setInputCol(inputColumn)
      .setOutputCol(outputColumn)
      .setVocabSize(vocabSize)
      .setMinDF(minDF)
    this
  }

  def fit(data: DataFrame): CountVectorizerTask = {
    countVectorizerModel = countVectorizer.fit(data)
    this
  }

  def transform(data: DataFrame): CountVectorizerTask = {
    transform = countVectorizerModel.transform(data)
    this
  }

  def getTransform(): DataFrame = {
    transform
  }

  def loadCountVectorizerModel(spark: SparkSession, path: String): CountVectorizerTask = {
    countVectorizerModel = CountVectorizerModel.load(path)
    this
  }

}
