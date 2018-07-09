package fr.toxic.spark.classification.crossValidation

import fr.toxic.spark.classification.task.{CrossValidationModelFactory, GbtClassifierTask}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame


class CrossValidationGbtClassifierTask(override val data: DataFrame,
                                       override val labelColumn: String,
                                       override val featureColumn: String,
                                       override val predictionColumn: String,
                                       override val pathModel: String,
                                       override val pathPrediction: String) extends CrossValidationTask(data, labelColumn, featureColumn, predictionColumn, pathModel, pathPrediction) with CrossValidationModelFactory {

  var estimator: GBTClassifier = _

  override def run(): CrossValidationGbtClassifierTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineCrossValidatorModel()
    fit()
    this
  }

  override def defineEstimator(): CrossValidationGbtClassifierTask = {
    estimator = new GbtClassifierTask(labelColumn=labelColumn,
                                      featureColumn=featureColumn,
                                      predictionColumn=predictionColumn).defineModel.getModel
    this
  }

  override def defineGridParameters(): CrossValidationGbtClassifierTask = {
      paramGrid = new ParamGridBuilder()
        .addGrid(estimator.maxDepth, Array(4, 8, 16, 30))
        .addGrid(estimator.maxBins, Array(2, 4, 8, 16))
        .build()
    this
  }

  override def defineCrossValidatorModel(): CrossValidationGbtClassifierTask = {
    crossValidator = new CrossValidator()
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEstimator(estimator)
    this
  }

  def getEstimator: GBTClassifier = {
    estimator
  }

  def getBestModel: GBTClassificationModel = {
    crossValidatorModel.bestModel.asInstanceOf[GBTClassificationModel]
  }

}