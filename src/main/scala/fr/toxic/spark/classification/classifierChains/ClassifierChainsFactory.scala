package fr.toxic.spark.classification.binaryRelevance

import org.apache.spark.sql.DataFrame

trait ClassifierChainsFactory {

  def run(data: DataFrame): ClassifierChainsFactory

}