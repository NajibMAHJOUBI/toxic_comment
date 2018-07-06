package fr.toxic.spark.classification.binaryRelevance

import fr.toxic.spark.classification.crossValidation.CrossValidationDecisionTreeTask
import fr.toxic.spark.classification.task.DecisionTreeTask
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 13/06/18.
  */
class BinaryRelevanceDecisionTreeTask(val columns: Array[String],
                                      val savePath: String,
                                      val featureColumn: String = "tf_idf",
                                      val methodValidation: String = "simple") {

  var prediction: DataFrame = _
  var model: DecisionTreeClassificationModel = _

  def run(data: DataFrame): Unit = {
    prediction = data
    columns.foreach(column => {
      val labelFeatures = BinaryRelevanceObject.createLabel(prediction, column)
      computeModel(labelFeatures, column)
      saveModel(column)
      prediction = computePrediction(labelFeatures)
    })
    BinaryRelevanceObject.savePrediction(prediction, columns, s"$savePath/prediction")
    BinaryRelevanceObject.multiLabelPrecision(prediction, columns)
  }

  def computeModel(data: DataFrame, column: String): Unit = {
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
  }

  def computePrediction(data: DataFrame): DataFrame = {
    model.transform(data).drop(Seq("rawPrediction", "probability"): _*)
  }

  def saveModel(column: String): Unit = {
    model.write.overwrite().save(s"$savePath/model/$column")
  }

  def loadModel(path: String): BinaryRelevanceDecisionTreeTask = {
    val decisionTree = new DecisionTreeTask(featureColumn = "tf_idf")
    model = decisionTree.loadModel(path).getModelFit
    this
  }

//  def getPrediction: DataFrame = {
//    prediction
//  }

}
