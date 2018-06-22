package fr.toxic.spark

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 12/06/18.
  */
class LogisticRegressionTask(val labelColumn: String = "label",
                             val featureColumn: String = "features",
                             val predictionColumn: String = "prediction") {


  var model: LogisticRegression = _
  var modelFit: LogisticRegressionModel = _
  var prediction: DataFrame = _

  def getPrediction(): DataFrame = {
    prediction
  }

  def getModel(): LogisticRegression = {
    model
  }

  def defineModel(): LogisticRegressionTask= {
    model = new LogisticRegression()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    this
  }

  def fit(data: DataFrame): LogisticRegressionTask = {
    modelFit = getModel().fit(data)
    this
  }

  def transform(data: DataFrame): LogisticRegressionTask = {
    prediction = modelFit.transform(data)
    this
  }

  def saveModel(path: String): LogisticRegressionTask = {
    model.save(path)
    this
  }

  def setRegParam(value: Double): LogisticRegressionTask = {
    model.setRegParam(value)
    this
  }

  def getRegParam(): Double = {
    model.getRegParam
  }
}
