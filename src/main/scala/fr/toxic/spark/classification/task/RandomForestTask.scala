package fr.toxic.spark.classification.task

import org.apache.spark.ml.classification.{RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.sql.DataFrame

class RandomForestTask(val labelColumn: String = "label",
                       val featureColumn: String = "features",
                       val predictionColumn: String = "prediction") extends ClassificationModelFactory {

  var model: RandomForestClassifier = _
  var modelFit: RandomForestClassificationModel = _
  var transform: DataFrame = _

  override def defineModel: RandomForestTask= {
    model = new RandomForestClassifier()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    this
  }

  override def fit(data: DataFrame): RandomForestTask = {
    modelFit = getModel.fit(data)
    this
  }

  def getModel: RandomForestClassifier = {
    model
  }

  override def transform(data: DataFrame): RandomForestTask = {
    transform = modelFit.transform(data)
    this
  }

  override def loadModel(path: String): RandomForestTask = {
    modelFit = RandomForestClassificationModel.load(path)
    this
  }

  override def saveModel(path: String): RandomForestTask = {
    model.save(path)
    this
  }

  def getModelFit: RandomForestClassificationModel = {
    modelFit
  }

  def getTransform: DataFrame = {
    transform
  }

  def setMaxDepth(value: Int): RandomForestTask = {
    model.setMaxDepth(value)
    this
  }

  def setMaxBins(value: Int): RandomForestTask = {
    model.setMaxBins(value)
    this
  }

  def getMaxBins: Int = {
    model.getMaxBins
  }

  def getMaxDepth: Int = {
    model.getMaxDepth
  }
}
