package fr.toxic.spark.classification.crossValidation

import fr.toxic.spark.classification.task.{CrossValidationModelFactory, LogisticRegressionTask}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.DataFrame


class CrossValidationLogisticRegressionTask(override val data: DataFrame,
                                            override val labelColumn: String,
                                            override val featureColumn: String,
                                            override val predictionColumn: String,
                                            override val pathModel: String,
                                            override val pathPrediction: String) extends CrossValidationTask(data, labelColumn, featureColumn, predictionColumn, pathModel, pathPrediction) with CrossValidationModelFactory {

  var estimator: LogisticRegression = _

  override def run(): CrossValidationLogisticRegressionTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineCrossValidatorModel()
    fit()
    this
  }

  override def defineEstimator(): CrossValidationLogisticRegressionTask = {
    estimator = new LogisticRegressionTask(labelColumn=labelColumn,
                                           featureColumn=featureColumn,
                                           predictionColumn=predictionColumn).defineModel.getModel
    this
  }

  override def defineGridParameters(): CrossValidationLogisticRegressionTask = {
      paramGrid = new ParamGridBuilder()
        .addGrid(estimator.regParam, Array(0.0, 0.001, 0.01, 0.1, 1.0, 10.0))
        .addGrid(estimator.elasticNetParam, Array(0.0, 0.25, 0.5, 0.75, 1.0))
        .build()
    this
  }

  override def defineCrossValidatorModel(): CrossValidationLogisticRegressionTask = {
    crossValidator = new CrossValidator()
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEstimator(estimator)
    this
  }

  def getEstimator: LogisticRegression = {
    estimator
  }

  def getBestModel: LogisticRegressionModel = {
    crossValidatorModel.bestModel.asInstanceOf[LogisticRegressionModel]
  }

}