package fr.toxic.spark.classification.multiLabelClassification.binaryRelevance

import fr.toxic.spark.classification.crossValidation.CrossValidationRandomForestTask
import fr.toxic.spark.classification.multiLabelClassification.{MultiLabelClassificationFactory, MultiLabelClassificationObject}
import fr.toxic.spark.classification.task.RandomForestTask
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 13/06/18.
  *
  * Binary relevance method based on random forest classifier
  *
  */

class BinaryRelevanceRandomForestTask(override val columns: Array[String],
                                      override val savePath: String,
                                      override val featureColumn: String, override val methodValidation: String) extends BinaryRelevanceTask(columns, savePath, featureColumn, methodValidation) with MultiLabelClassificationFactory {

  var model: RandomForestClassificationModel = _

  override def run(data: DataFrame): BinaryRelevanceRandomForestTask = {
    prediction = data
    columns.foreach(label => {
      val labelFeatures = MultiLabelClassificationObject.createLabel(prediction, label)
      computeModel(labelFeatures, label)
      saveModel(label)
      computePrediction(labelFeatures)
    })
    MultiLabelClassificationObject.savePrediction(prediction, columns, s"$savePath/prediction")
    MultiLabelClassificationObject.multiLabelPrecision(prediction, columns)
    this
  }

  override def computeModel(data: DataFrame, column: String): BinaryRelevanceRandomForestTask = {
    model = if (methodValidation == "cross_validation") {
      val cv = new CrossValidationRandomForestTask(data = data,
                                                   labelColumn = s"label_$column",
                                                   featureColumn = featureColumn,
                                                   predictionColumn = s"prediction_$column",
                                                   pathModel = "",
                                                   pathPrediction = "")
      cv.run()
      cv.getBestModel
    } else {
      val randomForest = new RandomForestTask(labelColumn = s"label_$column", featureColumn=featureColumn,
        predictionColumn = s"prediction_$column")
      randomForest.defineModel
      randomForest.fit(data)
      randomForest.getModelFit
    }
    this
  }

  override def computePrediction(data: DataFrame): BinaryRelevanceRandomForestTask = {
    prediction = model.transform(data).drop(Seq("rawPrediction", "probability"): _*)
    this
  }

  override def saveModel(column: String): BinaryRelevanceRandomForestTask = {
    model.write.overwrite().save(s"$savePath/model/$column")
    this
  }

  override def loadModel(path: String): BinaryRelevanceRandomForestTask = {
    model = new RandomForestTask(featureColumn = featureColumn).loadModel(path).getModelFit
    this
  }

}
