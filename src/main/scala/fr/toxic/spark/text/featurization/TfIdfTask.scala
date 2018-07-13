package fr.toxic.spark.text.featurization

import org.apache.spark.ml.feature.{IDF, IDFModel}
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 12/06/18.
  */
class TfIdfTask(val inputColumn: String = "tf", val outputColumn: String = "tf_idf") extends TextFeaturizationTask with TextFeaturizationFactory {

  private var idf: IDF = _
  private var idfModel: IDFModel = _

  override def run(data: DataFrame): TfIdfTask = {
    defineModel()
    fit(data)
    transform(data)
    this
  }

  override def defineModel(): TfIdfTask = {
    idf = new IDF().setInputCol(inputColumn).setOutputCol(outputColumn)
    this
  }

  override def fit(data: DataFrame): TfIdfTask = {
    idfModel = idf.fit(data)
    this
  }

  override def transform(data: DataFrame): TfIdfTask = {
    prediction = idfModel.transform(data)
    this
  }

  override def loadModel(path: String): TfIdfTask = {
    idfModel = IDFModel.load(path)
    this
  }

}
