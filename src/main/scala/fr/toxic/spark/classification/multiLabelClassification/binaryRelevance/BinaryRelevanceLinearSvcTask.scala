package fr.toxic.spark.classification.task.binaryRelevance

import fr.toxic.spark.classification.multiLabelClassification.binaryRelevance.{BinaryRelevanceFactory, BinaryRelevanceObject, BinaryRelevanceTask}
import fr.toxic.spark.classification.crossValidation.CrossValidationLinearSvcTask
import fr.toxic.spark.classification.multiLabelClassification.MultiLabelObject
import fr.toxic.spark.classification.task.LinearSvcTask
import org.apache.spark.ml.classification.LinearSVCModel
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 13/06/18.
  */
class BinaryRelevanceLinearSvcTask(override val columns: Array[String],
                                   override val savePath: String,
                                   override val featureColumn: String,
                                   override val methodValidation: String) extends
  BinaryRelevanceTask(columns, savePath, featureColumn, methodValidation) with BinaryRelevanceFactory {

  var model: LinearSVCModel = _

  override def run(data: DataFrame): BinaryRelevanceLinearSvcTask = {
    prediction = data
    columns.foreach(column => {
      val labelFeatures = BinaryRelevanceObject.createLabel(prediction, column)
      computeModel(labelFeatures, column)
      saveModel(column)
      computePrediction(labelFeatures)
    })
    MultiLabelObject.savePrediction(prediction, columns, s"$savePath/prediction")
    MultiLabelObject.multiLabelPrecision(prediction, columns)
    this
  }

  override def computeModel(data: DataFrame, column: String): BinaryRelevanceLinearSvcTask = {
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
    this
  }

  override def computePrediction(data: DataFrame): BinaryRelevanceLinearSvcTask = {
    prediction = model.transform(data).drop(Seq("rawPrediction", "probability"): _*)
    this
  }

  def loadModel(path: String): BinaryRelevanceLinearSvcTask = {
    val logisticRegression = new LinearSvcTask(featureColumn = "tf_idf")
    model = logisticRegression.loadModel(path).getModelFit
    this
  }

  override def saveModel(column: String): BinaryRelevanceLinearSvcTask = {
    model.write.overwrite().save(s"$savePath/model/$column")
    this
  }

}
