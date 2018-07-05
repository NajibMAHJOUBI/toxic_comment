package fr.toxic.spark.classification.task.binaryRelevance

import fr.toxic.spark.classification.binaryRelevance.BinaryRelevanceObject
import fr.toxic.spark.classification.crossValidation.CrossValidationLogisticRegressionTask
import fr.toxic.spark.classification.task.LogisticRegressionTask
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.mllib.evaluation.MultilabelMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.ml.linalg.Vector

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
    columns.map(column => {
      val labelFeatures = BinaryRelevanceObject.createLabel(prediction, column)
      val model = computeModel(labelFeatures, column)
      saveModel(column)
      if (probability) {
        prediction = computeProbability(labelFeatures, column)
      } else {
        prediction = computePrediction(labelFeatures)
      }
    })
    savePrediction(prediction)
    multiLabelPrecision(prediction)
  }

  def computeModel(data: DataFrame, column: String): Unit = {
    if (methodValidation == "cross_validation") {
      val cv = new CrossValidationLogisticRegressionTask(data = data, labelColumn = s"label_$column",
                                                         featureColumn = featureColumn,
                                                         predictionColumn = s"prediction_$column", pathModel = "",
                                                         pathPrediction = "")
      cv.run()
      model = cv.getBestModel
    } else{
      val logisticRegression = new LogisticRegressionTask(labelColumn = s"label_$column", featureColumn=featureColumn,
        predictionColumn = s"prediction_$column")
      logisticRegression.defineModel
      logisticRegression.fit(data)
      model = logisticRegression.getModelFit
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

  def multiLabelPrecision(data: DataFrame): Unit= {
    val predictionColumns = columns.map(column => s"prediction_$column")
    val labelColumns = columns.map(column => s"label_$column")
    val predictionAndLabels: RDD[(Array[Double], Array[Double])] =
      data.rdd.map(p => (predictionColumns.map(column => p.getDouble(p.fieldIndex(column))),
                         labelColumns.map(column => p.getLong(p.fieldIndex(column)).toDouble)))

    val metrics = new MultilabelMetrics(predictionAndLabels)
//    println(s"Recall = ${metrics.recall}")
//    println(s"Precision = ${metrics.precision}")
//    println(s"F1 measure = ${metrics.f1Measure}")
    println(s"Accuracy = ${metrics.accuracy}")
  }

  def savePrediction(data: DataFrame): Unit = {
    val columnsToKeep: Set[Column] = (Set("id")
      ++ columns.map(name => s"label_$name").toSet
      ++ columns.map(name => s"prediction_$name").toSet
      )
      .map(name => col(name))

    data
      .select(columnsToKeep.toSeq: _*)
      .write.option("header", "true").mode("overwrite")
      .csv(s"$savePath/prediction")
  }

  def loadModel(path: String): BinaryRelevanceLogisticRegressionTask = {
    val logisticRegression = new LogisticRegressionTask(featureColumn = "tf_idf")
    model = logisticRegression.loadModel(path).getModelFit
    this
  }

  def saveModel(column: String): Unit = {
    model.write.overwrite().save(s"$savePath/model/$column")
  }

  def getPrediction(): DataFrame = {
    prediction
  }

}
