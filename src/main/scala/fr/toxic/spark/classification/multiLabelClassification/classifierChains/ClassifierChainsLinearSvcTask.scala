package fr.toxic.spark.classification.multiLabelClassification.classifierChains

import fr.toxic.spark.classification.multiLabelClassification.binaryRelevance.ClassifierChainsFactory
import fr.toxic.spark.classification.crossValidation.CrossValidationLinearSvcTask
import fr.toxic.spark.classification.task.LinearSvcTask
import org.apache.spark.ml.classification.LinearSVCModel
import org.apache.spark.sql.DataFrame


class ClassifierChainsLinearSvcTask(override val labelColumns: Array[String],
                                    override val featureColumn: String,
                                    override val methodValidation: String,
                                    override val savePath: String) extends ClassifierChainsTask(labelColumns, featureColumn,
  methodValidation, savePath) with ClassifierChainsFactory {

  var model: LinearSVCModel = _

  override def run(data: DataFrame): ClassifierChainsLinearSvcTask = {
    labelColumns.map(label => {
      val newData = createNewDataSet(data, label: String)
      if (methodValidation == "cross_validation") {
        val cv = new CrossValidationLinearSvcTask(data = newData, labelColumn = label,
          featureColumn = featureColumn,
          predictionColumn = s"prediction_$label", pathModel = "",
          pathPrediction = "")
        cv.run()
        cv.getBestModel.write.overwrite().save(s"$savePath/$label")
      } else {
        val linearSvc = new LinearSvcTask(labelColumn = label, featureColumn = featureColumn,
          predictionColumn = s"prediction_$label")
        linearSvc.defineModel
        linearSvc.fit(newData)
        linearSvc.saveModel(s"$savePath/$label")
      }
    })
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