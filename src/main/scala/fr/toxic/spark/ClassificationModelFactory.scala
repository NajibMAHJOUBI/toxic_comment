package fr.toxic.spark

import org.apache.spark.sql.DataFrame

trait ClassificationModelFactory {

  def defineModel(): ClassificationModelFactory

  def fit(data: DataFrame): ClassificationModelFactory

  def transform(data: DataFrame): ClassificationModelFactory

}
