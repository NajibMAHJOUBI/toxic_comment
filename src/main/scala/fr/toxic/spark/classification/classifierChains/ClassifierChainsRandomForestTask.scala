package fr.toxic.spark.classification.classifierChains

import fr.toxic.spark.classification.binaryRelevance.ClassifierChainsFactory
import fr.toxic.spark.classification.crossValidation.CrossValidationRandomForestTask
import fr.toxic.spark.classification.task.RandomForestTask
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.sql.DataFrame


class ClassifierChainsRandomForestTask(override val labelColumns: Array[String],
                                       override val featureColumn: String,
                                       override val methodValidation: String,
                                       override val savePath: String) extends ClassifierChainsTask(labelColumns, featureColumn,
  methodValidation, savePath) with ClassifierChainsFactory {

  var model: RandomForestClassificationModel = _

  override def run(data: DataFrame): ClassifierChainsRandomForestTask = {
    labelColumns.map(label => {
      val newData = createNewDataSet(data, label: String)
      if (methodValidation == "cross_validation") {
        val cv = new CrossValidationRandomForestTask(data = newData, labelColumn = label,
          featureColumn = featureColumn,
          predictionColumn = s"prediction_$label", pathModel = "",
          pathPrediction = "")
        cv.run()
        cv.getBestModel.write.overwrite().save(s"$savePath/$label")
      } else {
        val randomForest = new RandomForestTask(labelColumn = label, featureColumn=featureColumn,
          predictionColumn = s"prediction_$label")
        randomForest.defineModel
        randomForest.fit(newData)
        randomForest.saveModel(s"$savePath/$label")
      }
    })
    this
  }

  override def computePrediction(data: DataFrame): ClassifierChainsRandomForestTask = {
    prediction = model.transform(data).drop(Seq("rawPrediction", "probability"): _*)
    this
  }

  override def saveModel(column: String): ClassifierChainsRandomForestTask = {
    model.write.overwrite().save(s"$savePath/model/$column")
    this
  }

  override def loadModel(path: String): ClassifierChainsRandomForestTask = {
    model = new RandomForestTask(featureColumn = featureColumn).loadModel(path).getModelFit
    this
  }

}