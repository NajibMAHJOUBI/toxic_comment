package fr.toxic.spark.classification.task

import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 12/06/18.
  */
class GbtClassifierTask(val labelColumn: String = "label",
                        val featureColumn: String = "features",
                        val predictionColumn: String = "prediction") extends ClassificationModelFactory {


  var model: GBTClassifier = _
  var modelFit: GBTClassificationModel = _
  var transform: DataFrame = _

  def getModelFit: GBTClassificationModel = {
    modelFit
  }

  override def defineModel(): GbtClassifierTask= {
    model = new GBTClassifier()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    this
  }

  override def fit(data: DataFrame): GbtClassifierTask = {
    modelFit = getModel.fit(data)
    this
  }

  def getModel: GBTClassifier = {
    model
  }

  override def transform(data: DataFrame): GbtClassifierTask = {
    transform = modelFit.transform(data)
    this
  }

  override def saveModel(path: String): GbtClassifierTask = {
    model.save(path)
    this
  }

  override def loadModel(path: String): GbtClassifierTask = {
    modelFit = GBTClassificationModel.load(path)
    this
  }

  def setMaxDepth(value: Int): GbtClassifierTask = {
    model.setMaxDepth(value)
    this
  }

  def getMaxDepth: Int = {
    model.getMaxDepth
  }

  def getTransform: DataFrame = {
    transform
  }
}
