package fr.toxic.spark

import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.sql.DataFrame

class CrossValidatorTask(val classificationModel: String) {

  def run(data: DataFrame): Unit = {

  }

  def defineCrossValidator: CrossValidatorModel = {
    CrossValidator(estimator=None, estimatorParamMaps=grid, evaluator=evaluator,
    parallelism=2)
  }

  def getEstimator()
}
