package fr.toxic.spark.text.featurization

import org.apache.spark.ml.feature.{IDF, IDFModel}
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 12/06/18.
  */
class TfIdfTask(val inputColumn: String = "tf", val outputColumn: String = "tf_idf") {

  private var idf: IDF = _
  private var idfModel: IDFModel = _
  private var transform: DataFrame = _

  def run(data: DataFrame): Unit = {
    defineModel()
    fit(data)
    transform(data)
  }

  def defineModel(): TfIdfTask = {
    idf = new IDF().setInputCol(inputColumn).setOutputCol(outputColumn)
    this
  }

  def fit(data: DataFrame): TfIdfTask = {
    idfModel = idf.fit(data)
    this
  }

  def transform(data: DataFrame): TfIdfTask = {
    transform = idfModel.transform(data)
    this
  }

  def getTransform(): DataFrame = {
    transform
  }
}
