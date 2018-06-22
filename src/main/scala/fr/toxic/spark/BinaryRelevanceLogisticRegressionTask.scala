package fr.toxic.spark

import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions.col

/**
  * Created by mahjoubi on 13/06/18.
  */
class BinaryRelevanceLogisticRegressionTask(val columns: Array[String], val savePath: String,
                                            val featureColumn: String = "tf_idf",
                                            val methodValidation: String = "simple") {

  var prediction: DataFrame = _

  def run(data: DataFrame): Unit = {
    var prediction: DataFrame = data
    columns.map(column => {
      val labelFeatures = createLabel(prediction, column)
      prediction = computeModel(labelFeatures, column)
    })
    savePrediction(prediction)
  }

//  def createFeatures(data: DataFrame): DataFrame = {
//    data.withColumnRenamed(featureColumn, "features")
//  }

  def createLabel(data: DataFrame, column: String): DataFrame = {
    data.withColumnRenamed(column, s"label_${column}")
  }

  def computeModel(data: DataFrame, column: String): DataFrame = {
    val logisticRegression = new LogisticRegressionTask(labelColumn = s"label_${column}", featureColumn=featureColumn,
      predictionColumn = s"prediction_${column}")
    logisticRegression.fit(data)
    logisticRegression.transform(data)
    logisticRegression.getPrediction().drop("probability").drop("rawPrediction")
  }

//  def computeModelValidation(dataFrame: DataFrame): Dataframe = {
//    if (methodValidation == "simple") {
//
//    }
//  }

  def savePrediction(data: DataFrame): Unit = {
    val columnsToKeep: Set[Column] = (Set("id")
      ++ columns.map(name => s"label_${name}").toSet
      ++ columns.map(name => s"prediction_${name}").toSet
      )
      .map(name => col(name))

    data
      .select(columnsToKeep.toSeq: _*)
      .write.option("header", "true").mode("overwrite")
      .csv(s"${savePath}/prediction")
  }

}
