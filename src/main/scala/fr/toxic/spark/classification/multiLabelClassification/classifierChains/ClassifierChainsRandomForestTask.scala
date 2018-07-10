package fr.toxic.spark.classification.multiLabelClassification.classifierChains

import fr.toxic.spark.classification.crossValidation.CrossValidationRandomForestTask
import fr.toxic.spark.classification.multiLabelClassification.MultiLabelObject
import fr.toxic.spark.classification.multiLabelClassification.binaryRelevance.ClassifierChainsFactory
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

  override def computeModel(data: DataFrame, column: String): ClassifierChainsRandomForestTask = {
    model = if (methodValidation == "cross_validation") {
      val cv = new CrossValidationRandomForestTask(data = data, labelColumn = column,
        featureColumn = featureColumn,
        predictionColumn = s"prediction_$column", pathModel = "",
        pathPrediction = "")
      cv.run()
      cv.getBestModel
    } else {
      val randomForest = new RandomForestTask(labelColumn = column, featureColumn=featureColumn,
        predictionColumn = s"prediction_$column")
      randomForest.defineModel
      randomForest.fit(data)
      randomForest.getModelFit
    }
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