package fr.toxic.spark.classification.multiLabelClassification.binaryRelevance

import fr.toxic.spark.classification.crossValidation.CrossValidationDecisionTreeTask
import fr.toxic.spark.classification.task.DecisionTreeTask
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 13/06/18.
  *
  * Binary Relevance with Decision Tree Classifier
  *
  */
class BinaryRelevanceDecisionTreeTask(override val columns: Array[String],
                                      override val savePath: String,
                                      override val featureColumn: String,
                                      override val methodValidation: String) extends
  BinaryRelevanceTask(columns, savePath, featureColumn, methodValidation) with BinaryRelevanceFactory {

  var model: DecisionTreeClassificationModel = _

  override def run(data: DataFrame): BinaryRelevanceDecisionTreeTask = {
    prediction = data
    columns.foreach(column => {
      val labelFeatures = BinaryRelevanceObject.createLabel(prediction, column)
      computeModel(labelFeatures, column)
      saveModel(column)
      computePrediction(labelFeatures)
    })
    BinaryRelevanceObject.savePrediction(prediction, columns, s"$savePath/prediction")
    BinaryRelevanceObject.multiLabelPrecision(prediction, columns)
    this
  }

  override def computeModel(data: DataFrame, column: String): BinaryRelevanceDecisionTreeTask = {
    model = if (methodValidation == "cross_validation") {
      val cv = new CrossValidationDecisionTreeTask(data = data, labelColumn = s"label_$column",
                                                   featureColumn = featureColumn,
                                                   predictionColumn = s"prediction_$column", pathModel = "",
                                                   pathPrediction = "")
      cv.run()
      cv.getBestModel
    } else{
      val decisionTree = new DecisionTreeTask(labelColumn = s"label_$column", featureColumn=featureColumn,
        predictionColumn = s"prediction_$column")
      decisionTree.defineModel
      decisionTree.fit(data)
      decisionTree.getModelFit
    }
    this
  }

  override def computePrediction(data: DataFrame): BinaryRelevanceDecisionTreeTask = {
    prediction = model.transform(data).drop(Seq("rawPrediction", "probability"): _*)
    this
  }

  override def saveModel(column: String): BinaryRelevanceDecisionTreeTask = {
    model.write.overwrite().save(s"$savePath/model/$column")
    this
  }

  override def loadModel(path: String): BinaryRelevanceDecisionTreeTask = {
    val decisionTree = new DecisionTreeTask(featureColumn = "tf_idf")
    model = decisionTree.loadModel(path).getModelFit
    this
  }

}
