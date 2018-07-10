package fr.toxic.spark.classification.multiLabelClassification.classifierChains

import fr.toxic.spark.classification.multiLabelClassification.binaryRelevance.ClassifierChainsFactory
import fr.toxic.spark.classification.crossValidation.CrossValidationGbtClassifierTask
import fr.toxic.spark.classification.task.GbtClassifierTask
import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.sql.DataFrame


class ClassifierChainsGbtClassifierTask(override val labelColumns: Array[String],
                                        override val featureColumn: String,
                                        override val methodValidation: String,
                                        override val savePath: String) extends ClassifierChainsTask(labelColumns, featureColumn,
  methodValidation, savePath) with ClassifierChainsFactory {

  var model: GBTClassificationModel = _

  override def run(data: DataFrame): ClassifierChainsGbtClassifierTask = {
    labelColumns.map(label => {
      val newData = createNewDataSet(data, label: String)
      if (methodValidation == "cross_validation") {
        val cv = new CrossValidationGbtClassifierTask(data = newData, labelColumn = label,
          featureColumn = featureColumn,
          predictionColumn = s"prediction_$label", pathModel = "",
          pathPrediction = "")
        cv.run()
        cv.getBestModel.write.overwrite().save(s"$savePath/$label")
      } else {
        val gbtClassifier = new GbtClassifierTask(labelColumn = label, featureColumn = featureColumn,
          predictionColumn = s"prediction_$label")
        gbtClassifier.defineModel
        gbtClassifier.fit(newData)
        gbtClassifier.saveModel(s"$savePath/$label")
      }
    })
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