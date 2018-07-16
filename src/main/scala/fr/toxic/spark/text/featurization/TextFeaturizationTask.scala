package fr.toxic.spark.text.featurization

import org.apache.spark.sql.DataFrame

class TextFeaturizationTask {

  var prediction: DataFrame = _

  def getPrediction: DataFrame = {
    prediction
  }

}
