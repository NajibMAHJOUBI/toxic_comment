package fr.toxic.spark.classification.stackingMethod

import fr.toxic.spark.classification.crossValidation.CrossValidationLogisticRegressionTask
import fr.toxic.spark.classification.task.LogisticRegressionTask
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.sql.{DataFrame, SparkSession}


class StackingMethodLogisticRegressionTask(override val labels: Array[String],
                                           override val classificationMethods: Array[String],
                                           override val pathLabel: String, override val pathPrediction: String, override val pathSave: String) extends StackingMethodTask(labels, classificationMethods, pathLabel, pathPrediction, pathSave) with StackingMethodFactory {

  val labelColumn: String = "label"
  val featureColumn: String = "features"
  var model: LogisticRegressionModel = _

  override def run(spark: SparkSession): StackingMethodLogisticRegressionTask = {
    labels.foreach(label => {
      mergeData(spark, label)
      val data = createLabelFeatures(spark, label)
      computeModel(data, label)
      saveModel(label)
    })

    this
  }

  override def computeModel(data: DataFrame, label: String): StackingMethodLogisticRegressionTask = {
    val cv = new CrossValidationLogisticRegressionTask(data = data, labelColumn = label,
      featureColumn = featureColumn,
      predictionColumn = s"prediction_$label", pathModel = "",
      pathPrediction = "")
    cv.run()
    model = cv.getBestModel
    this
  }

  override def saveModel(label: String): StackingMethodLogisticRegressionTask = {
    model.write.overwrite().save(s"$pathSave/model/$label")
    this
  }

  override def loadModel(path: String): StackingMethodLogisticRegressionTask = {
    model = new LogisticRegressionTask(featureColumn = featureColumn).loadModel(path).getModelFit
    this
  }

  override def computePrediction(data: DataFrame): StackingMethodLogisticRegressionTask = {
    prediction = model.transform(data).drop(Seq("rawPrediction", "probability"): _*)
    this
  }

}
