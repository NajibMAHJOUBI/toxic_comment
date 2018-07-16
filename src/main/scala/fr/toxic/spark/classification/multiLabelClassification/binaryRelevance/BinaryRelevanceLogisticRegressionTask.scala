package fr.toxic.spark.classification.task.binaryRelevance

import fr.toxic.spark.classification.crossValidation.CrossValidationLogisticRegressionTask
import fr.toxic.spark.classification.multiLabelClassification.binaryRelevance.BinaryRelevanceTask
import fr.toxic.spark.classification.multiLabelClassification.{MultiLabelClassificationFactory, MultiLabelClassificationObject}
import fr.toxic.spark.classification.task.LogisticRegressionTask
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 13/06/18.
  *
  * Binary relevance method based on random forest classifier
  *
  */
class BinaryRelevanceLogisticRegressionTask(override val columns: Array[String],
                                            override val savePath: String,
                                            override val featureColumn: String, override val methodValidation: String) extends BinaryRelevanceTask(columns, savePath, featureColumn, methodValidation) with MultiLabelClassificationFactory {

  var model: LogisticRegressionModel = _

  override def run(data: DataFrame): BinaryRelevanceLogisticRegressionTask = {
    prediction = data
    columns.foreach(column => {
      val labelFeatures = MultiLabelClassificationObject.createLabel(prediction, column)
      computeModel(labelFeatures, column)
      saveModel(column)
      computePrediction(labelFeatures)
    })
    MultiLabelClassificationObject.savePrediction(prediction, columns, s"$savePath/prediction")
    MultiLabelClassificationObject.multiLabelPrecision(prediction, columns)
    this
  }

  override def computeModel(data: DataFrame, column: String): BinaryRelevanceLogisticRegressionTask = {
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
    this
  }

  override def computePrediction(data: DataFrame): BinaryRelevanceLogisticRegressionTask = {
    prediction = model.transform(data).drop(Seq("rawPrediction", "probability"): _*)
    this
  }

  override def loadModel(path: String): BinaryRelevanceLogisticRegressionTask = {
    model = new LogisticRegressionTask(featureColumn = "tf_idf").loadModel(path).getModelFit
    this
  }

  override def saveModel(column: String): BinaryRelevanceLogisticRegressionTask = {
    model.write.overwrite().save(s"$savePath/model/$column")
    this
  }

}
