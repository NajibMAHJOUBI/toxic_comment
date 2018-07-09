package fr.toxic.spark.classification.classifierChains

import fr.toxic.spark.classification.binaryRelevance.ClassifierChainsFactory
import fr.toxic.spark.classification.crossValidation.CrossValidationDecisionTreeTask
import fr.toxic.spark.classification.task.DecisionTreeTask
import org.apache.spark.sql.DataFrame


class ClassifierChainsDecisionTreeTask(override val labelColumns: Array[String],
                                       override val featureColumn: String,
                                       override val methodValidation: String,
                                       override val savePath: String) extends ClassifierChainsTask(labelColumns, featureColumn,
  methodValidation, savePath) with ClassifierChainsFactory {

  override def run(data: DataFrame): ClassifierChainsDecisionTreeTask = {
    labelColumns.map(label => {
      val newData = createNewDataSet(data, label: String)
      if (methodValidation == "cross_validation") {
        val cv = new CrossValidationDecisionTreeTask(data = newData, labelColumn = label,
          featureColumn = featureColumn,
          predictionColumn = s"prediction_$label", pathModel = "",
          pathPrediction = "")
        cv.run()
        cv.getBestModel.write.overwrite().save(s"$savePath/$label")
      } else {
        val decisionTree = new DecisionTreeTask(labelColumn = label, featureColumn = featureColumn,
          predictionColumn = s"prediction_$label")
        decisionTree.defineModel
        decisionTree.fit(newData)
        decisionTree.saveModel(s"$savePath/$label")
      }
    })
    this
  }

}