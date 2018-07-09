package fr.toxic.spark.classification.crossValidation

import fr.toxic.spark.classification.task.{CrossValidationModelFactory, DecisionTreeTask}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.DataFrame


class CrossValidationDecisionTreeTask(override val data: DataFrame,
                                      override val labelColumn: String,
                                      override val featureColumn: String,
                                      override val predictionColumn: String,
                                      override val pathModel: String,
                                      override val pathPrediction: String) extends CrossValidationTask(data, labelColumn, featureColumn, predictionColumn, pathModel, pathPrediction) with CrossValidationModelFactory {

  var estimator: DecisionTreeClassifier = _

  override def run(): CrossValidationDecisionTreeTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineCrossValidatorModel()
    fit()
    this
  }

  override def defineEstimator(): CrossValidationDecisionTreeTask = {
    estimator = new DecisionTreeTask(labelColumn=labelColumn,
                                     featureColumn=featureColumn,
                                     predictionColumn=predictionColumn).defineModel.getModel
    this
  }

  override def defineGridParameters(): CrossValidationDecisionTreeTask = {
      paramGrid = new ParamGridBuilder()
        .addGrid(estimator.maxDepth, Array(4, 8, 16, 30))
        .addGrid(estimator.maxBins, Array(2, 4, 8, 16))
        .build()
    this
  }

  override def defineCrossValidatorModel(): CrossValidationDecisionTreeTask = {
    crossValidator = new CrossValidator()
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEstimator(estimator)
    this
  }

  def getEstimator: DecisionTreeClassifier = {
    estimator
  }

  def getBestModel: DecisionTreeClassificationModel = {
    crossValidatorModel.bestModel.asInstanceOf[DecisionTreeClassificationModel]
  }

}