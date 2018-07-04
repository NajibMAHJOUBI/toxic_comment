package fr.toxic.spark.classification.crossValidation

import fr.toxic.spark.classification.task.DecisionTreeTask
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.DataFrame


class CrossValidationDecisionTreeTask(val data: DataFrame,
                                      val labelColumn: String,
                                      val featureColumn: String,
                                      val predictionColumn: String,
                                      val pathModel: String,
                                      val pathPrediction: String) {

  var estimator: DecisionTreeClassifier = _
  var evaluator: BinaryClassificationEvaluator = _
  var paramGrid: Array[ParamMap] = _
  var crossValidator: CrossValidator = _
  var crossValidatorModel: CrossValidatorModel = _

  def run(): CrossValidationDecisionTreeTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineCrossValidatorModel()
    fit()
  }

  def defineEstimator(): CrossValidationDecisionTreeTask = {
    estimator = new DecisionTreeTask(labelColumn=labelColumn,
                                     featureColumn=featureColumn,
                                     predictionColumn=predictionColumn).defineModel.getModel
    this
  }

  def defineGridParameters(): CrossValidationDecisionTreeTask = {
      paramGrid = new ParamGridBuilder()
        .addGrid(estimator.maxDepth, Array(4, 8, 16, 30))
        .addGrid(estimator.maxBins, Array(2, 4, 8, 16))
        .build()
    this
  }

  def defineEvaluator(): CrossValidationDecisionTreeTask = {
      evaluator = new BinaryClassificationEvaluator()
        .setRawPredictionCol(predictionColumn)
        .setLabelCol(labelColumn)
        .setMetricName("areaUnderROC")
    this}

  def defineCrossValidatorModel(): CrossValidationDecisionTreeTask = {
    crossValidator = new CrossValidator()
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEstimator(estimator)
    this
  }

  def fit(): CrossValidationDecisionTreeTask = {
    crossValidatorModel = crossValidator.fit(data)
    this
  }

  def transform(data: DataFrame): DataFrame = {
    crossValidatorModel.transform(data)
  }

  def getLabelColumn: String = {
    labelColumn
  }

  def getFeatureColumn: String = {
    featureColumn
  }

  def getPredictionColumn: String = {
    predictionColumn
  }

  def getGridParameters: Array[ParamMap] = {
    paramGrid
  }

  def getEstimator: DecisionTreeClassifier = {
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

  def getBestModel: DecisionTreeClassificationModel = {
    crossValidatorModel.bestModel.asInstanceOf[DecisionTreeClassificationModel]
  }

  def setGridParameters(grid: Array[ParamMap]): CrossValidationDecisionTreeTask = {
     paramGrid = grid
     this
   }
}