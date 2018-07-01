package fr.toxic.spark.classification.task.binaryRelevance

import fr.toxic.spark.classification.crossValidation.CrossValidationLinearSvcTask
import fr.toxic.spark.classification.task.LinearSvcTask
import org.apache.spark.ml.classification.LinearSVCModel
import org.apache.spark.mllib.evaluation.MultilabelMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{Column, DataFrame}

/**
  * Created by mahjoubi on 13/06/18.
  */
class BinaryRelevanceLinearSvcTask(val data: DataFrame,
                                   val columns: Array[String], val savePath: String,
                                   val featureColumn: String = "tf_idf",
                                   val methodValidation: String = "simple") {

  var prediction: DataFrame = data
  var model: LinearSVCModel = _

  def run(): Unit = {
    columns.map(column => {
      val labelFeatures = createLabel(prediction, column)
      val model = computeModel(labelFeatures, column)
      saveModel(column)
      prediction = computePrediction(labelFeatures, model)
    })
    savePrediction(prediction)
    multiLabelPrecision(prediction)
  }

  def createLabel(data: DataFrame, column: String): DataFrame = {
    data.withColumnRenamed(column, s"label_$column")
  }

  def computeModel(data: DataFrame, column: String): LinearSVCModel = {
    if (methodValidation == "cross_validation") {
      val cv = new CrossValidationLinearSvcTask(data = data, labelColumn = s"label_$column",
                                                         featureColumn = featureColumn,
                                                         predictionColumn = s"prediction_$column", pathModel = "",
                                                         pathPrediction = "")
      cv.run()
      model = cv.getBestModel()
    } else{
      val linearSvc = new LinearSvcTask(labelColumn = s"label_$column", featureColumn=featureColumn,
        predictionColumn = s"prediction_$column")
      linearSvc.defineModel()
      linearSvc.fit(data)
      model = linearSvc.getModelFit
    }
    model
  }

  def computePrediction(data: DataFrame, model: LinearSVCModel): DataFrame = {
    model.transform(data).drop(Seq("rawPrediction", "probability"): _*)
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

  def saveModel(column: String): Unit = {
    model.write.overwrite().save(s"$savePath/model/$column")
  }

  def getPrediction(): DataFrame = {
    prediction
  }

}
