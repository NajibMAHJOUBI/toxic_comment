package fr.toxic.spark.classification.task

trait CrossValidationModelFactory {

  def run(): CrossValidationModelFactory

  def defineEstimator(): CrossValidationModelFactory

  def defineGridParameters(): CrossValidationModelFactory

  def defineCrossValidatorModel(): CrossValidationModelFactory

  def getLabelColumn: String

  def getFeatureColumn: String

  def getPredictionColumn: String

}
