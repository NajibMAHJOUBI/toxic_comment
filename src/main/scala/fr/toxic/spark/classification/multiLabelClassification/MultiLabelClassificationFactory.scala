package fr.toxic.spark.classification.multiLabelClassification

import org.apache.spark.sql.DataFrame

trait MultiLabelClassificationFactory {

  def run(data: DataFrame): MultiLabelClassificationFactory

  def computeModel(data: DataFrame, column: String): MultiLabelClassificationFactory

  def computePrediction(data: DataFrame): MultiLabelClassificationFactory

  def saveModel(column: String): MultiLabelClassificationFactory

  def loadModel(path: String): MultiLabelClassificationFactory

}