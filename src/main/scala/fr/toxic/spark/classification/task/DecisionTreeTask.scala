package fr.toxic.spark.classification.task

import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.sql.DataFrame

class DecisionTreeTask (val labelColumn: String = "label",
                        val featureColumn: String = "features",
                        val predictionColumn: String = "prediction") extends ClassificationModelFactory {

  var model: DecisionTreeClassifier = _
  var modelFit: DecisionTreeClassificationModel = _
  var transform: DataFrame = _

  override def defineModel(): DecisionTreeTask= {
    model = new DecisionTreeClassifier()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    this
  }

  override def fit(data: DataFrame): DecisionTreeTask = {
    modelFit = getModel.fit(data)
    this
  }

  def getModel: DecisionTreeClassifier = {
    model
  }

  override def transform(data: DataFrame): DecisionTreeTask = {
    transform = modelFit.transform(data)
    this
  }

  override def loadModel(path: String): DecisionTreeTask = {
    modelFit = DecisionTreeClassificationModel.load(path)
    this
  }

  override def saveModel(path: String): DecisionTreeTask = {
    model.save(path)
    this
  }

  def getModelFit: DecisionTreeClassificationModel = {
    modelFit
  }

  def getTransform: DataFrame = {
    transform
  }

  def setMaxDepth(value: Int): DecisionTreeTask = {
    model.setMaxDepth(value)
    this
  }

  def setMaxBins(value: Int): DecisionTreeTask = {
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
