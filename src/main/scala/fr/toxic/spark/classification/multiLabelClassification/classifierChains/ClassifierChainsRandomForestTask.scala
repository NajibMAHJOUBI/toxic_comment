package fr.toxic.spark.classification.multiLabelClassification.classifierChains

import fr.toxic.spark.classification.crossValidation.CrossValidationRandomForestTask
import fr.toxic.spark.classification.multiLabelClassification.{MultiLabelClassificationFactory, MultiLabelClassificationObject}
import fr.toxic.spark.classification.task.RandomForestTask
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.sql.DataFrame


class ClassifierChainsRandomForestTask(override val labelColumns: Array[String],
                                       override val featureColumn: String,
                                       override val methodValidation: String,
                                       override val savePath: String) extends ClassifierChainsTask(labelColumns, featureColumn,
  methodValidation, savePath) with MultiLabelClassificationFactory {

  var model: RandomForestClassificationModel = _

  override def run(data: DataFrame): ClassifierChainsRandomForestTask = {
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