package fr.toxic.spark.classification.multiLabelClassification.classifierChains

import fr.toxic.spark.classification.crossValidation.CrossValidationLogisticRegressionTask
import fr.toxic.spark.classification.multiLabelClassification.MultiLabelObject
import fr.toxic.spark.classification.multiLabelClassification.binaryRelevance.ClassifierChainsFactory
import fr.toxic.spark.classification.task.LogisticRegressionTask
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.sql.DataFrame


class ClassifierChainsLogisticRegressionTask(override val labelColumns: Array[String],
                                             override val featureColumn: String,
                                             override val methodValidation: String = "simple",
                                             override val savePath: String) extends ClassifierChainsTask(labelColumns, featureColumn,
  methodValidation, savePath) with ClassifierChainsFactory {

  var model: LogisticRegressionModel = _

  override def run(data: DataFrame): ClassifierChainsLogisticRegressionTask = {
    prediction = data
    labelColumns.map(label => {
      val newData = createNewDataSet(data, label)
      computeModel(newData, label)
      saveModel(label)
      computePrediction(newData)
    })
    MultiLabelObject.savePrediction(prediction, labelColumns, s"$savePath/prediction")
    MultiLabelObject.multiLabelPrecision(prediction, labelColumns)
    this
  }

  override def computeModel(data: DataFrame, column: String): ClassifierChainsLogisticRegressionTask = {
    model = if (methodValidation == "cross_validation") {
      val cv = new CrossValidationLogisticRegressionTask(data = data, labelColumn = column,
        featureColumn = featureColumn,
        predictionColumn = s"prediction_$column", pathModel = "",
        pathPrediction = "")
      cv.run()
      cv.getBestModel
    } else {
      val logisticRegression = new LogisticRegressionTask(labelColumn = column, featureColumn=featureColumn,
        predictionColumn = s"prediction_$column")
      logisticRegression.defineModel
      logisticRegression.fit(data)
      logisticRegression.getModelFit
    }
    this
  }

  override def computePrediction(data: DataFrame): ClassifierChainsLogisticRegressionTask = {
    prediction = model.transform(data).drop(Seq("rawPrediction", "probability"): _*)
    this
  }

  override def saveModel(column: String): ClassifierChainsLogisticRegressionTask = {
    model.write.overwrite().save(s"$savePath/model/$column")
    this
  }

  override def loadModel(path: String): ClassifierChainsLogisticRegressionTask = {
    model = new LogisticRegressionTask(featureColumn = featureColumn).loadModel(path).getModelFit
    this
  }

}