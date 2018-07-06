package fr.toxic.spark.classification.task.binaryRelevance

import fr.toxic.spark.classification.binaryRelevance.BinaryRelevanceObject
import fr.toxic.spark.classification.crossValidation.CrossValidationLogisticRegressionTask
import fr.toxic.spark.classification.task.LogisticRegressionTask
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

/**
  * Created by mahjoubi on 13/06/18.
  */
class BinaryRelevanceLogisticRegressionTask(val columns: Array[String], val savePath: String,
                                            val featureColumn: String = "tf_idf",
                                            val methodValidation: String = "simple",
                                            val probability: Boolean = false) {
  var prediction: DataFrame = _
  var model: LogisticRegressionModel = _

  def run(data: DataFrame): Unit = {
    prediction = data
    columns.foreach(column => {
      val labelFeatures = BinaryRelevanceObject.createLabel(prediction, column)
      computeModel(labelFeatures, column)
      saveModel(column)
      prediction = if (probability) {computeProbability(labelFeatures, column)} else {computePrediction(labelFeatures)}
    })
    BinaryRelevanceObject.savePrediction(prediction, columns, s"$savePath/prediction")
    BinaryRelevanceObject.multiLabelPrecision(prediction, columns)
  }

  def computeModel(data: DataFrame, column: String): Unit = {
    model = if (methodValidation == "cross_validation") {
      val cv = new CrossValidationLogisticRegressionTask(data = data, labelColumn = s"label_$column",
                                                         featureColumn = featureColumn,
                                                         predictionColumn = s"prediction_$column", pathModel = "",
                                                         pathPrediction = "")
      cv.run()
      cv.getBestModel
    } else {
      val logisticRegression = new LogisticRegressionTask(labelColumn = s"label_$column", featureColumn=featureColumn,
        predictionColumn = s"prediction_$column")
      logisticRegression.defineModel
      logisticRegression.fit(data)
      logisticRegression.getModelFit
    }
  }

  def computePrediction(data: DataFrame): DataFrame = {
    model.transform(data).drop(Seq("rawPrediction", "probability"): _*)
  }

  def computeProbability(data: DataFrame, label: String): DataFrame = {
    val getProbability = udf((probability: Vector, prediction: Double) => BinaryRelevanceObject.getPredictionProbability(probability, prediction))
    val transform = model.transform(data).withColumn("predictionProbability", getProbability(col("probability"), col(s"prediction_$label")))
    transform.drop(Seq("rawPrediction", "probability", s"prediction_$label"): _*).withColumnRenamed("predictionProbability", s"prediction_$label")
  }

  def loadModel(path: String): BinaryRelevanceLogisticRegressionTask = {
    val logisticRegression = new LogisticRegressionTask(featureColumn = "tf_idf")
    model = logisticRegression.loadModel(path).getModelFit
    this
  }

  def saveModel(column: String): Unit = {
    model.write.overwrite().save(s"$savePath/model/$column")
  }

  def getPrediction: DataFrame = {
    prediction
  }

}
