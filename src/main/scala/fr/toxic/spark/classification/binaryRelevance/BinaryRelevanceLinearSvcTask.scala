package fr.toxic.spark.classification.task.binaryRelevance

import fr.toxic.spark.classification.binaryRelevance.BinaryRelevanceObject
import fr.toxic.spark.classification.crossValidation.CrossValidationLinearSvcTask
import fr.toxic.spark.classification.task.LinearSvcTask
import org.apache.spark.ml.classification.LinearSVCModel
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 13/06/18.
  */
class BinaryRelevanceLinearSvcTask(val columns: Array[String],
                                   val savePath: String,
                                   val featureColumn: String = "tf_idf",
                                   val methodValidation: String = "simple") {

  var prediction: DataFrame = _
  var model: LinearSVCModel = _

  def run(data: DataFrame): Unit = {
    prediction = data
    columns.foreach(column => {
      val labelFeatures = BinaryRelevanceObject.createLabel(prediction, column)
      computeModel(labelFeatures, column)
      saveModel(column)
      prediction = computePrediction(labelFeatures)
    })
    BinaryRelevanceObject.savePrediction(prediction, columns, s"$savePath/prediction")
    BinaryRelevanceObject.multiLabelPrecision(prediction, columns)
  }

  def computeModel(data: DataFrame, column: String): Unit = {
    model = if (methodValidation == "cross_validation") {
      val cv = new CrossValidationLinearSvcTask(data = data, labelColumn = s"label_$column",
                                                         featureColumn = featureColumn,
                                                         predictionColumn = s"prediction_$column", pathModel = "",
                                                         pathPrediction = "")
      cv.run()
      cv.getBestModel
    } else {
      val linearSvc = new LinearSvcTask(labelColumn = s"label_$column", featureColumn=featureColumn,
        predictionColumn = s"prediction_$column")
      linearSvc.defineModel
      linearSvc.fit(data)
      linearSvc.getModelFit
    }
  }

  def computePrediction(data: DataFrame): DataFrame = {
    model.transform(data).drop(Seq("rawPrediction", "probability"): _*)
  }

  def loadModel(path: String): BinaryRelevanceLinearSvcTask = {
    val logisticRegression = new LinearSvcTask(featureColumn = "tf_idf")
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
