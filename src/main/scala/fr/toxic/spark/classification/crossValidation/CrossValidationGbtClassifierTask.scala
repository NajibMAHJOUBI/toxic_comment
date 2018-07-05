package fr.toxic.spark.classification.crossValidation

import fr.toxic.spark.classification.task.{ClassificationModelFactory, GbtClassifierTask}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.DataFrame


class CrossValidationGbtClassifierTask(val data: DataFrame,
                                       val labelColumn: String,
                                       val featureColumn: String,
                                       val predictionColumn: String,
                                       val pathModel: String,
                                       val pathPrediction: String) extends ClassificationModelFactory {

  var estimator: GBTClassifier = _
  var evaluator: BinaryClassificationEvaluator = _
  var paramGrid: Array[ParamMap] = _
  var crossValidator: CrossValidator = _
  var crossValidatorModel: CrossValidatorModel = _

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

  override def defineEvaluator(): CrossValidationGbtClassifierTask = {
      evaluator = new BinaryClassificationEvaluator()
        .setRawPredictionCol(predictionColumn)
        .setLabelCol(labelColumn)
        .setMetricName("areaUnderROC")
    this}

  override def defineCrossValidatorModel(): CrossValidationGbtClassifierTask = {
    crossValidator = new CrossValidator()
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEstimator(estimator)
    this
  }

  override def fit(): CrossValidationGbtClassifierTask = {
    crossValidatorModel = crossValidator.fit(data)
    this
  }

  override def transform(data: DataFrame): DataFrame = {
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

  def getEstimator: GBTClassifier = {
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

  def getBestModel: GBTClassificationModel = {
    crossValidatorModel.bestModel.asInstanceOf[GBTClassificationModel]
  }

  def setGridParameters(grid: Array[ParamMap]): CrossValidationGbtClassifierTask = {
     paramGrid = grid
     this
   }
}