package fr.toxic.spark.classification.binaryRelevance


import fr.toxic.spark.classification.crossValidation.CrossValidationRandomForestTask
import fr.toxic.spark.classification.task.RandomForestTask
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 13/06/18.
  */
class BinaryRelevanceRandomForestTask(val columns: Array[String], val savePath: String,
                                      val featureColumn: String = "tf_idf",
                                      val methodValidation: String = "simple") extends BinaryRelevanceTask {

  var model: RandomForestClassificationModel = _

  def run(data: DataFrame): Unit = {
    prediction = data
    columns.foreach(column => {
      val labelFeatures = BinaryRelevanceObject.createLabel(prediction, column)
      computeModel(labelFeatures, column)
      saveModel(column)
      computePrediction(labelFeatures)
    })
    BinaryRelevanceObject.savePrediction(prediction, columns, s"$savePath/prediction")
    BinaryRelevanceObject.multiLabelPrecision(prediction, columns)
  }

  def computeModel(data: DataFrame, column: String): BinaryRelevanceRandomForestTask = {
    model = if (methodValidation == "cross_validation") {
      val cv = new CrossValidationRandomForestTask(data = data, labelColumn = s"label_$column",
                                                   featureColumn = featureColumn,
                                                   predictionColumn = s"prediction_$column", pathModel = "",
                                                   pathPrediction = "")
      cv.run()
      cv.getBestModel
    } else{
      val decisionTree = new RandomForestTask(labelColumn = s"label_$column", featureColumn=featureColumn,
        predictionColumn = s"prediction_$column")
      decisionTree.defineModel
      decisionTree.fit(data)
      decisionTree.getModelFit
    }
    this
  }

  def computePrediction(data: DataFrame): BinaryRelevanceRandomForestTask = {
    prediction = model.transform(data).drop(Seq("rawPrediction", "probability"): _*)
    this
  }

  def loadModel(path: String): BinaryRelevanceRandomForestTask = {
    val logisticRegression = new RandomForestTask(featureColumn = "tf_idf")
    model = logisticRegression.loadModel(path).getModelFit
    this
  }

  def saveModel(column: String): Unit = {
    model.write.overwrite().save(s"$savePath/model/$column")
  }

}
