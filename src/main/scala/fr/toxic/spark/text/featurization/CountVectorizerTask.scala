package fr.toxic.spark.text.featurization

import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 12/06/18.
  */
class CountVectorizerTask(val inputColumn: String = "words", val outputColumn: String = "tf", val minDF: Int = 2, val vocabSize: Int = 3) extends TextFeaturizationTask with TextFeaturizationFactory {

  private var countVectorizer: CountVectorizer = _
  private var countVectorizerModel: CountVectorizerModel = _

  override def run(data: DataFrame): CountVectorizerTask = {
    defineModel()
    fit(data)
    transform(data)
  }

  override def defineModel(): CountVectorizerTask = {
    countVectorizer =  new CountVectorizer()
      .setInputCol(inputColumn)
      .setOutputCol(outputColumn)
      .setVocabSize(vocabSize)
      .setMinDF(minDF)
    this
  }

  override def fit(data: DataFrame): CountVectorizerTask = {
    countVectorizerModel = countVectorizer.fit(data)
    this
  }

  override def transform(data: DataFrame): CountVectorizerTask = {
    prediction = countVectorizerModel.transform(data)
    this
  }

  override def loadModel(path: String): CountVectorizerTask = {
    countVectorizerModel = CountVectorizerModel.load(path)
    this
  }

}
