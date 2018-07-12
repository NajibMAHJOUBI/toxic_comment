package fr.toxic.spark.classification.stackingMethod

import org.apache.spark.sql.SparkSession

class StackingMethodLogisticRegressionTask(override val labels: Array[String],
                                           override val classificationMethods: Array[String],
                                           override val pathLabel: String,
                                           override val pathPrediction: String)
  extends StackingMethodTask(labels, classificationMethods, pathLabel, pathPrediction) with StackingMethodFactory {

  override def run(spark: SparkSession): StackingMethodLogisticRegressionTask = {
    labels.foreach(label => {
      val data = mergeData(spark, label)})
      data.show()
    this
  }

}
