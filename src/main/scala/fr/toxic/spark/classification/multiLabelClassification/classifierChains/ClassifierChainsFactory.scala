package fr.toxic.spark.classification.multiLabelClassification.binaryRelevance

import org.apache.spark.sql.DataFrame

trait ClassifierChainsFactory {

  def run(data: DataFrame): ClassifierChainsFactory

  def computeModel(data: DataFrame, column: String): ClassifierChainsFactory

  def computePrediction(data: DataFrame): ClassifierChainsFactory

  def saveModel(column: String): ClassifierChainsFactory

  def loadModel(path: String): ClassifierChainsFactory

}