package fr.toxic.spark.classification.task

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 12/06/18.
  */
class LogisticRegressionTask(val labelColumn: String = "label",
                             val featureColumn: String = "features",
                             val predictionColumn: String = "prediction") extends ClassificationModelFactory {


  var model: LogisticRegression = _
  var modelFit: LogisticRegressionModel = _
  var transform: DataFrame = _

  def getModelFit: LogisticRegressionModel = {
    modelFit
  }

  override def defineModel(): LogisticRegressionTask= {
    model = new LogisticRegression()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    this
  }

  override def fit(data: DataFrame): LogisticRegressionTask = {
    modelFit = getModel.fit(data)
    this
  }

  def getModel: LogisticRegression = {
    model
  }

  override def transform(data: DataFrame): LogisticRegressionTask = {
    transform = modelFit.transform(data)
    this
  }

  override def saveModel(path: String): LogisticRegressionTask = {
    model.save(path)
    this
  }

  def setRegParam(value: Double): LogisticRegressionTask = {
    model.setRegParam(value)
    this
  }

  def getRegParam: Double = {
    model.getRegParam
  }

  override def loadModel(path: String): LogisticRegressionTask = {
    modelFit = LogisticRegressionModel.load(path)
    this
  }

  def getTransform: DataFrame = {
    transform
  }
}
