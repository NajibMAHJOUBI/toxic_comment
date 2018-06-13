package fr.toxic.spark

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 12/06/18.
  */
class LogisticRegressionTask(val labelColumn: String = "label", val featureColumn: String = "features") {

  var model: LogisticRegressionModel = _
  var prediction: DataFrame = _

  def fitModel(data: DataFrame): LogisticRegressionTask = {
    model = new LogisticRegression().setFeaturesCol(featureColumn).setLabelCol(labelColumn).fit(data)
    this
  }

  def transformModel(data: DataFrame): LogisticRegressionTask = {
    prediction = model.transform(data)
    this
  }

  def getPrediction(): DataFrame = {
    prediction
  }

}
