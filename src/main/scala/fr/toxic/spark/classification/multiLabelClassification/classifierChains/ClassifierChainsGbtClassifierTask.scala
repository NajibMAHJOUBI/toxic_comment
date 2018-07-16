package fr.toxic.spark.classification.multiLabelClassification.classifierChains

import fr.toxic.spark.classification.crossValidation.CrossValidationGbtClassifierTask
import fr.toxic.spark.classification.multiLabelClassification.{MultiLabelClassificationFactory, MultiLabelClassificationObject}
import fr.toxic.spark.classification.task.GbtClassifierTask
import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.sql.DataFrame


class ClassifierChainsGbtClassifierTask(override val labelColumns: Array[String],
                                        override val featureColumn: String,
                                        override val methodValidation: String, override val savePath: String) extends ClassifierChainsTask(labelColumns, featureColumn, methodValidation, savePath) with MultiLabelClassificationFactory {

  var model: GBTClassificationModel = _

  override def run(data: DataFrame): ClassifierChainsGbtClassifierTask = {
    prediction = data
    labelColumns.map(label => {
      val labelFeatures = createNewDataSet(prediction, label)
      computeModel(labelFeatures, label)
      saveModel(label)
      computePrediction(labelFeatures)
    })
    labelColumns.foreach(label => prediction = MultiLabelClassificationObject.createLabel(prediction, label))
    MultiLabelClassificationObject.savePrediction(prediction, labelColumns, s"$savePath/prediction")
    MultiLabelClassificationObject.multiLabelPrecision(prediction, labelColumns)
    this
  }

  override def computeModel(data: DataFrame, column: String): ClassifierChainsGbtClassifierTask = {
    model = if (methodValidation == "cross_validation") {
      val cv = new CrossValidationGbtClassifierTask(data = data, labelColumn = column,
        featureColumn = featureColumn,
        predictionColumn = s"prediction_$column", pathModel = "",
        pathPrediction = "")
      cv.run()
      cv.getBestModel
    } else {
      val gbtClassifier = new GbtClassifierTask(labelColumn = column, featureColumn=featureColumn,
        predictionColumn = s"prediction_$column")
      gbtClassifier.defineModel
      gbtClassifier.fit(data)
      gbtClassifier.getModelFit
    }
    this
  }

  override def computePrediction(data: DataFrame): ClassifierChainsGbtClassifierTask = {
    prediction = model.transform(data).drop(Seq("rawPrediction", "probability"): _*)
    this
  }

  override def saveModel(column: String): ClassifierChainsGbtClassifierTask = {
    model.write.overwrite().save(s"$savePath/model/$column")
    this
  }

  override def loadModel(path: String): ClassifierChainsGbtClassifierTask = {
    model = new GbtClassifierTask(featureColumn = featureColumn).loadModel(path).getModelFit
    this
  }

}