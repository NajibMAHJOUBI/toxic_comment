package fr.toxic.spark.classification.crossValidation

import fr.toxic.spark.classification.task.{CrossValidationModelFactory, DecisionTreeTask}
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
                                      val pathPrediction: String) extends CrossValidationModelFactory {

  var estimator: DecisionTreeClassifier = _
  var evaluator: BinaryClassificationEvaluator = _
  var paramGrid: Array[ParamMap] = _
  var crossValidator: CrossValidator = _
  var crossValidatorModel: CrossValidatorModel = _

  override def run(): CrossValidationDecisionTreeTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineCrossValidatorModel()
    fit()
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

  override def defineEvaluator(): CrossValidationDecisionTreeTask = {
      evaluator = new BinaryClassificationEvaluator()
        .setRawPredictionCol(predictionColumn)
        .setLabelCol(labelColumn)
        .setMetricName("areaUnderROC")
    this}

  override def defineCrossValidatorModel(): CrossValidationDecisionTreeTask = {
    crossValidator = new CrossValidator()
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEstimator(estimator)
    this
  }

  override def fit(): CrossValidationDecisionTreeTask = {
    crossValidatorModel = crossValidator.fit(data)
    this
  }

  override def transform(data: DataFrame): DataFrame = {
    crossValidatorModel.transform(data)
  }

  override def getLabelColumn: String = {
    labelColumn
  }

  override def getFeatureColumn: String = {
    featureColumn
  }

  override def getPredictionColumn: String = {
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