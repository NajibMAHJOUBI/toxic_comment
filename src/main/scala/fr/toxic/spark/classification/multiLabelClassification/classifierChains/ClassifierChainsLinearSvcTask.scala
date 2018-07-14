package fr.toxic.spark.classification.multiLabelClassification.classifierChains

import fr.toxic.spark.classification.crossValidation.CrossValidationLinearSvcTask
import fr.toxic.spark.classification.multiLabelClassification.{MultiLabelClassificationFactory, MultiLabelClassificationObject}
import fr.toxic.spark.classification.task.LinearSvcTask
import org.apache.spark.ml.classification.LinearSVCModel
import org.apache.spark.sql.DataFrame


class ClassifierChainsLinearSvcTask(override val labelColumns: Array[String],
                                    override val featureColumn: String,
                                    override val methodValidation: String,
                                    override val savePath: String) extends ClassifierChainsTask(labelColumns, featureColumn,
  methodValidation, savePath) with MultiLabelClassificationFactory {

  var model: LinearSVCModel = _

  override def run(data: DataFrame): ClassifierChainsLinearSvcTask = {
    prediction = data
    labelColumns.map(label => {
      val newData = createNewDataSet(data, label)
      computeModel(newData, label)
      saveModel(label)
      computePrediction(newData)
    })
    labelColumns.foreach(label => prediction = MultiLabelClassificationObject.createLabel(prediction, label))
    MultiLabelClassificationObject.savePrediction(prediction, labelColumns, s"$savePath/prediction")
    MultiLabelClassificationObject.multiLabelPrecision(prediction, labelColumns)
    this
  }

  override def computeModel(data: DataFrame, column: String): ClassifierChainsLinearSvcTask = {
    model = if (methodValidation == "cross_validation") {
      val cv = new CrossValidationLinearSvcTask(data = data, labelColumn = s"label_$column",
        featureColumn = featureColumn,
        predictionColumn = s"prediction_$column", pathModel = "",
        pathPrediction = "")
      cv.run()
      cv.getBestModel
    } else{
      val gbtClassifier = new LinearSvcTask(labelColumn = s"label_$column", featureColumn=featureColumn,
        predictionColumn = s"prediction_$column")
      gbtClassifier.defineModel
      gbtClassifier.fit(data)
      gbtClassifier.getModelFit
    }
    this
  }

  override def computePrediction(data: DataFrame): ClassifierChainsLinearSvcTask = {
    prediction = model.transform(data).drop(Seq("rawPrediction", "probability"): _*)
    this
  }

  override def saveModel(column: String): ClassifierChainsLinearSvcTask = {
    model.write.overwrite().save(s"$savePath/model/$column")
    this
  }

  override def loadModel(path: String): ClassifierChainsLinearSvcTask = {
    model = new LinearSvcTask(featureColumn = featureColumn).loadModel(path).getModelFit
    this
  }

}