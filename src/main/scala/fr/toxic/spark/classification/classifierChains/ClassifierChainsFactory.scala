package fr.toxic.spark.classification.binaryRelevance

import org.apache.spark.sql.DataFrame

trait ClassifierChainsFactory {

  def run(data: DataFrame): ClassifierChainsFactory

  def computePrediction(data: DataFrame): ClassifierChainsFactory

  def saveModel(column: String): ClassifierChainsFactory

  def loadModel(path: String): ClassifierChainsFactory

}