package fr.toxic.spark.classification.task

import org.apache.spark.sql.DataFrame

trait CrossValidationModelFactory {

  def run(): CrossValidationModelFactory

  def defineEstimator(): CrossValidationModelFactory

  def defineGridParameters(): CrossValidationModelFactory

  def defineEvaluator(): CrossValidationModelFactory

  def defineCrossValidatorModel(): CrossValidationModelFactory

  def fit(): CrossValidationModelFactory

  def transform(data: DataFrame): CrossValidationModelFactory

}