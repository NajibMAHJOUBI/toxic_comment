package fr.toxic.spark.classification.binaryRelevance

import fr.toxic.spark.classification.crossValidation.CrossValidationGbtClassifierTask
import fr.toxic.spark.classification.task.GbtClassifierTask
import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 13/06/18.
  */

class BinaryRelevanceGbtClassifierTask(val columns: Array[String],
                                       val savePath: String,
                                       val featureColumn: String = "tf_idf",
                                       val methodValidation: String = "simple") {

  var prediction: DataFrame = _
  var model: GBTClassificationModel = _

  def run(data: DataFrame): Unit = {
    prediction = data
    columns.foreach(column => {
      val labelFeatures = BinaryRelevanceObject.createLabel(prediction, column)
      computeModel(labelFeatures, column)
      saveModel(column)
      prediction = computePrediction(labelFeatures)
    })
    BinaryRelevanceObject.savePrediction(prediction, columns,s"$savePath/prediction")
    BinaryRelevanceObject.multiLabelPrecision(prediction, columns)
  }

  def createLabel(data: DataFrame, column: String): DataFrame = {
    data.withColumnRenamed(column, s"label_$column")
  }

  def computeModel(data: DataFrame, column: String): Unit = {
    model = if (methodValidation == "cross_validation") {
      val cv = new CrossValidationGbtClassifierTask(data = data, labelColumn = s"label_$column",
                                                    featureColumn = featureColumn,
                                                    predictionColumn = s"prediction_$column", pathModel = "",
                                                    pathPrediction = "")
      cv.run()
      cv.getBestModel
    } else{
      val gbtClassifier = new GbtClassifierTask(labelColumn = s"label_$column", featureColumn=featureColumn,
                                               predictionColumn = s"prediction_$column")
      gbtClassifier.defineModel
      gbtClassifier.fit(data)
      gbtClassifier.getModelFit
    }
  }

  def computePrediction(data: DataFrame): DataFrame = {
    model.transform(data).drop(Seq("rawPrediction", "probability"): _*)
  }

  def loadModel(path: String): BinaryRelevanceGbtClassifierTask = {
    val gbtClassifier = new GbtClassifierTask(featureColumn = "tf_idf")
    model = gbtClassifier.loadModel(path).getModelFit
    this
  }

  def saveModel(column: String): Unit = {
    model.write.overwrite().save(s"$savePath/model/$column")
  }

  def getPrediction: DataFrame = {
    prediction
  }

}
