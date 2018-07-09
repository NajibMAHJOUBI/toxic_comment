package fr.toxic.spark.classification.binaryRelevance

import org.apache.spark.sql.DataFrame

trait BinaryRelevanceFactory {

  def run(data: DataFrame): BinaryRelevanceFactory

  def computeModel(data: DataFrame, column: String): BinaryRelevanceFactory

  def computePrediction(data: DataFrame): BinaryRelevanceFactory

  def saveModel(column: String): BinaryRelevanceFactory

  def loadModel(path: String): BinaryRelevanceFactory

}