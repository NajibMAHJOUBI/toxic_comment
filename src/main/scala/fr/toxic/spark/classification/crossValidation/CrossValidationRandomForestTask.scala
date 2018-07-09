package fr.toxic.spark.classification.crossValidation

import fr.toxic.spark.classification.task.{CrossValidationModelFactory, RandomForestTask}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame


class CrossValidationRandomForestTask(override val data: DataFrame,
                                      override val labelColumn: String,
                                      override val featureColumn: String,
                                      override val predictionColumn: String,
                                      override val pathModel: String,
                                      override val pathPrediction: String) extends CrossValidationTask(data, labelColumn, featureColumn, predictionColumn, pathModel, pathPrediction) with CrossValidationModelFactory {

  var estimator: RandomForestClassifier = _

  override def run(): CrossValidationRandomForestTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineCrossValidatorModel()
    fit()
    this
  }

  override def defineEstimator(): CrossValidationRandomForestTask = {
    estimator = new RandomForestTask(labelColumn=labelColumn,
                                     featureColumn=featureColumn,
                                     predictionColumn=predictionColumn).defineModel.getModel
    this
  }

  override def defineGridParameters(): CrossValidationRandomForestTask = {
      paramGrid = new ParamGridBuilder()
        .addGrid(estimator.maxDepth, Array(4, 8, 16, 30))
        .addGrid(estimator.maxBins, Array(2, 4, 8, 16))
        .build()
    this
  }

  override def defineCrossValidatorModel(): CrossValidationRandomForestTask = {
    crossValidator = new CrossValidator()
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEstimator(estimator)
    this
  }

  def getEstimator: RandomForestClassifier = {
    estimator
  }

  def getBestModel: RandomForestClassificationModel = {
    crossValidatorModel.bestModel.asInstanceOf[RandomForestClassificationModel]
  }

}