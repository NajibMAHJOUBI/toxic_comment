package fr.toxic.spark.text.featurization

import org.apache.spark.sql.DataFrame

trait TextFeaturizationFactory {

  def run(data: DataFrame): TextFeaturizationFactory

  def defineModel(): TextFeaturizationFactory

  def fit(data: DataFrame): TextFeaturizationFactory

  def transform(data: DataFrame): TextFeaturizationFactory

  def loadModel(path: String): TextFeaturizationFactory
}
