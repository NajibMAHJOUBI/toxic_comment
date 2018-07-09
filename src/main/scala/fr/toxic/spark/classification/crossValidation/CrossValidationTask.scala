package fr.toxic.spark.classification.crossValidation

import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.sql.DataFrame

class CrossValidationTask(val data: DataFrame,
                          val labelColumn: String,
                          val featureColumn: String,
                          val predictionColumn: String,
                          val pathModel: String,
                          val pathPrediction: String) {

  var evaluator: BinaryClassificationEvaluator = _
  var paramGrid: Array[ParamMap] = _
  var crossValidator: CrossValidator = _
  var crossValidatorModel: CrossValidatorModel = _

  def fit(): CrossValidationTask = {
    crossValidatorModel = crossValidator.fit(data)
    this
  }

  def transform(data: DataFrame): DataFrame = {
    crossValidatorModel.transform(data)
  }

  def defineEvaluator(): CrossValidationTask = {
    evaluator = new BinaryClassificationEvaluator()
      .setRawPredictionCol(predictionColumn)
      .setLabelCol(labelColumn)
      .setMetricName("areaUnderROC")
    this
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

  def getEvaluator: Evaluator = {
    evaluator
  }

  def getCrossValidator: CrossValidator = {
    crossValidator
  }

  def getCrossValidatorModel: CrossValidatorModel = {
    crossValidatorModel
  }


  def setGridParameters(grid: Array[ParamMap]): CrossValidationTask = {
    paramGrid = grid
    this
  }
  }
