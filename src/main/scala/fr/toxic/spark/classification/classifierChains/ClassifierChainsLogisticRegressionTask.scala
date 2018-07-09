package fr.toxic.spark.classification.classifierChains

import fr.toxic.spark.classification.binaryRelevance.ClassifierChainsFactory
import fr.toxic.spark.classification.crossValidation.CrossValidationLogisticRegressionTask
import fr.toxic.spark.classification.task.LogisticRegressionTask
import org.apache.spark.sql.DataFrame


class ClassifierChainsLogisticRegressionTask(override val labelColumns: Array[String],
                                             override val featureColumn: String,
                                             override val methodValidation: String = "simple",
                                             override val savePath: String) extends ClassifierChainsTask(labelColumns, featureColumn,
  methodValidation, savePath) with ClassifierChainsFactory {

  override def run(data: DataFrame): ClassifierChainsLogisticRegressionTask = {
    labelColumns.map(label => {
      val newData = createNewDataSet(data, label: String)
      if (methodValidation == "cross_validation") {
        val cv = new CrossValidationLogisticRegressionTask(data = newData, labelColumn = label,
          featureColumn = featureColumn,
          predictionColumn = s"prediction_$label", pathModel = "",
          pathPrediction = "")
        cv.run()
        cv.getBestModel.write.overwrite().save(s"$savePath/$label")
      } else {
        val logisticRegression = new LogisticRegressionTask(labelColumn = label, featureColumn=featureColumn,
          predictionColumn = s"prediction_$label")
        logisticRegression.defineModel
        logisticRegression.fit(newData)
        logisticRegression.saveModel(s"$savePath/$label")
      }
    })
    this
  }

}