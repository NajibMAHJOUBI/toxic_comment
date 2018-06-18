package fr.toxic.spark

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.DataFrame

class CrossValidationTask(val data: DataFrame,
                          val labelColumn: String,
                          val featureColumn: String,
                          val predictionColumn: String,
                          val modelClassifier: String) {

  var paramGrid: Array[ParamMap] = _
  var evaluator: BinaryClassificationEvaluator = _
  var crossValidator: CrossValidator = _
  var crossValidatorModel: CrossValidatorModel = _


  def run(): CrossValidationTask = {
    defineGridParameters()
    defineEvaluator()
    defineCrossValidatorModel()
    fitModel()
  }

  def defineGridParameters(): CrossValidationTask = {
    if (modelClassifier == "logistic_regression") {
      paramGrid = new ParamGridBuilder()
        .addGrid(new LogisticRegressionTask().getModel().regParam, Array(0.0, 0.001, 0.01, 0.1, 1.0, 10.0))
        .addGrid(new LogisticRegressionTask().getModel().elasticNetParam, Array(0.0, 0.25, 0.5, 0.75, 1.0))
        .build()
    }
    this
  }

  def defineEvaluator(): CrossValidationTask = {
      evaluator = new BinaryClassificationEvaluator()
        .setRawPredictionCol("prediction")
        .setLabelCol("label")
        .setMetricName("areaUnderROC")
    this}



  def defineCrossValidatorModel(): CrossValidationTask = {
    crossValidator = new CrossValidator().setEvaluator(evaluator).setEstimatorParamMaps(paramGrid)

    if (modelClassifier == "logistic_regression") {
      crossValidator = crossValidator.setEstimator(new LogisticRegressionTask().getModel())
    }

    this}

  def fitModel(): CrossValidationTask = {
    crossValidatorModel = crossValidator.fit(data)
    this}
}