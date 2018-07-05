package fr.toxic.spark.classification.crossValidation

import fr.toxic.spark.classification.task.{CrossValidationModelFactory, LogisticRegressionTask}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.DataFrame


class CrossValidationLogisticRegressionTask(val data: DataFrame,
                                            val labelColumn: String,
                                            val featureColumn: String,
                                            val predictionColumn: String,
                                            val pathModel: String,
                                            val pathPrediction: String) extends CrossValidationModelFactory {

  var estimator: LogisticRegression = _
  var evaluator: BinaryClassificationEvaluator = _
  var paramGrid: Array[ParamMap] = _
  var crossValidator: CrossValidator = _
  var crossValidatorModel: CrossValidatorModel = _

  override def run(): CrossValidationLogisticRegressionTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineCrossValidatorModel()
    fit()
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

  override def defineEvaluator(): CrossValidationLogisticRegressionTask = {
      evaluator = new BinaryClassificationEvaluator()
        .setRawPredictionCol(predictionColumn)
        .setLabelCol(labelColumn)
        .setMetricName("areaUnderROC")
    this}

  override def defineCrossValidatorModel(): CrossValidationLogisticRegressionTask = {
    crossValidator = new CrossValidator()
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEstimator(estimator)
    this
  }

  override def fit(): CrossValidationLogisticRegressionTask = {
    crossValidatorModel = crossValidator.fit(data)
    this
  }

  override def transform(data: DataFrame): DataFrame = {
    crossValidatorModel.transform(data)
  }

  override def getLabelColumn: String = {
    estimator.getLabelCol
  }

  override def getFeatureColumn: String = {
    estimator.getFeaturesCol
  }

  override def getPredictionColumn: String = {
    estimator.getPredictionCol
  }

  def getGridParameters: Array[ParamMap] = {
    paramGrid
  }

  def getEstimator: LogisticRegression = {
    estimator
  }

  def getEvaluator: Evaluator = {
    evaluator
  }

  def getCrossValidator: CrossValidator = {
    crossValidator
  }

  def getCrossValidatorModel: CrossValidatorModel = {
    crossValidatorModel
  }

  def getBestModel: LogisticRegressionModel = {
    crossValidatorModel.bestModel.asInstanceOf[LogisticRegressionModel]
  }

  def setGridParameters(grid: Array[ParamMap]): CrossValidationLogisticRegressionTask = {
     paramGrid = grid
     this
  }
}