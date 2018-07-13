package fr.toxic.spark.classification.stackingMethod

import fr.toxic.spark.classification.crossValidation.CrossValidationLogisticRegressionTask
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.sql.{DataFrame, SparkSession}

class StackingMethodLogisticRegressionTask(override val labels: Array[String],
                                           override val classificationMethods: Array[String],
                                           override val pathLabel: String,
                                           override val pathPrediction: String)
  extends StackingMethodTask(labels, classificationMethods, pathLabel, pathPrediction) with StackingMethodFactory {

  val labelColumn: String = "label"
  val featureColumn: String = "features"
  var model: LogisticRegressionModel = _

  override def run(spark: SparkSession): StackingMethodLogisticRegressionTask = {
    labels.foreach(label => {
      mergeData(spark, label)
      val data = createDfLabelFeatures(spark: SparkSession, label: String)
      computeModel(data, label)
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

}
