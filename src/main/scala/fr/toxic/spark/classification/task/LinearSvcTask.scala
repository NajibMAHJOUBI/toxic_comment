package fr.toxic.spark.classification.task

import org.apache.spark.ml.classification.{LinearSVC, LinearSVCModel}
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 12/06/18.
  *
  * LinearSVC classifier
  *
  */
class LinearSvcTask(val labelColumn: String = "label",
                    val featureColumn: String = "features",
                    val predictionColumn: String = "prediction") extends ClassificationModelFactory {


  var model: LinearSVC = _
  var modelFit: LinearSVCModel = _
  var transform: DataFrame = _

  def getModelFit(): LinearSVCModel = {
    modelFit
  }

  override def defineModel(): LinearSvcTask= {
    model = new LinearSVC()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    this
  }

  override def fit(data: DataFrame): LinearSvcTask = {
    modelFit = getModel().fit(data)
    this
  }

  def getModel(): LinearSVC = {
    model
  }

  override def transform(data: DataFrame): LinearSvcTask = {
    transform = modelFit.transform(data)
    this
  }

  def saveModel(path: String): LinearSvcTask = {
    model.save(path)
    this
  }

  def setRegParam(value: Double): LinearSvcTask = {
    model.setRegParam(value)
    this
  }

  def getRegParam(): Double = {
    model.getRegParam
  }

  def loadModel(path: String): LinearSvcTask = {
    modelFit = LinearSVCModel.load(path)
    this
  }

  def getTransform(): DataFrame = {
    transform
  }
}
