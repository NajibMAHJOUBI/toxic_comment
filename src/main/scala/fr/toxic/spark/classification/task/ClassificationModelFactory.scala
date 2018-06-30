package fr.toxic.spark.classification.task

import org.apache.spark.sql.DataFrame

trait ClassificationModelFactory {

  def defineModel(): ClassificationModelFactory

  def fit(data: DataFrame): ClassificationModelFactory

  def transform(data: DataFrame): ClassificationModelFactory

  def saveModel(path: String): ClassificationModelFactory

  def loadModel(path: String): ClassificationModelFactory

}
