package fr.toxic.spark.classification.multiLabelClassification.binaryRelevance

import fr.toxic.spark.classification.crossValidation.CrossValidationGbtClassifierTask
import fr.toxic.spark.classification.task.GbtClassifierTask
import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 13/06/18.
  *
  * Binary Relevance with GBT Classifier
  *
  */

class BinaryRelevanceGbtClassifierTask(override val columns: Array[String],
                                       override val savePath: String,
                                       override val featureColumn: String,
                                       override val methodValidation: String) extends
  BinaryRelevanceTask(columns, savePath, featureColumn, methodValidation) with BinaryRelevanceFactory {

  var model: GBTClassificationModel = _

  override def run(data: DataFrame): BinaryRelevanceGbtClassifierTask = {
    prediction = data
    columns.foreach(column => {
      val labelFeatures = BinaryRelevanceObject.createLabel(prediction, column)
      computeModel(labelFeatures, column)
      saveModel(column)
      computePrediction(labelFeatures)
    })
    BinaryRelevanceObject.savePrediction(prediction, columns,s"$savePath/prediction")
    BinaryRelevanceObject.multiLabelPrecision(prediction, columns)
    this
  }


  override def computeModel(data: DataFrame, column: String): BinaryRelevanceGbtClassifierTask = {
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
    this
  }

  override def computePrediction(data: DataFrame): BinaryRelevanceGbtClassifierTask = {
    prediction = model.transform(data).drop(Seq("rawPrediction", "probability"): _*)
    this
  }

  override def loadModel(path: String): BinaryRelevanceGbtClassifierTask = {
    model = new GbtClassifierTask(featureColumn = "tf_idf").loadModel(path).getModelFit
    this
  }

  override def saveModel(column: String): BinaryRelevanceGbtClassifierTask = {
    model.write.overwrite().save(s"$savePath/model/$column")
    this
  }

}
