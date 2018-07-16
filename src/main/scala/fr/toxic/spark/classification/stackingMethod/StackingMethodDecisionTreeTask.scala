package fr.toxic.spark.classification.stackingMethod

import fr.toxic.spark.classification.crossValidation.CrossValidationDecisionTreeTask
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.sql.{DataFrame, SparkSession}


class StackingMethodDecisionTreeTask(override val labels: Array[String],
                                     override val classificationMethods: Array[String],
                                     override val pathLabel: String,
                                     override val pathPrediction: String,
                                     override val pathSave: String)
  extends StackingMethodTask(labels, classificationMethods, pathLabel, pathPrediction, pathSave) with StackingMethodFactory {

  val featureColumn: String = "features"
  var model: DecisionTreeClassificationModel = _

  override def run(spark: SparkSession): StackingMethodDecisionTreeTask = {
    labels.foreach(label => {
      mergeData(spark, label)
      val data = createLabelFeatures(spark: SparkSession, label: String)
      computeModel(data, label)
      saveModel(label)
    })
    this
  }

  override def computeModel(data: DataFrame, label: String): StackingMethodDecisionTreeTask = {
    val cv = new CrossValidationDecisionTreeTask(data = data, labelColumn = label,
      featureColumn = featureColumn,
      predictionColumn = s"prediction_$label", pathModel = "",
      pathPrediction = "")
    cv.run()
    model = cv.getBestModel
    this
  }

  override def saveModel(column: String): StackingMethodDecisionTreeTask = {
    model.write.overwrite().save(s"$pathSave/model/$column")
    this
  }

  override def computePrediction(data: DataFrame): StackingMethodDecisionTreeTask = {
    this
  }

  def loadModel(path: String): StackingMethodDecisionTreeTask = {
    this
  }
}
