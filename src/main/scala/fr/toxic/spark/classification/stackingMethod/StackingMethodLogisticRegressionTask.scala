package fr.toxic.spark.classification.stackingMethod

import org.apache.spark.sql.SparkSession

class StackingMethodLogisticRegressionTask(override val labels: Array[String],
                                           override val classificationMethods: Array[String],
                                           override val pathLabel: String,
                                           override val pathPrediction: String)
  extends StackingMethodTask(labels, classificationMethods, pathLabel, pathPrediction) {

  def run(spark: SparkSession): StackingMethodLogisticRegressionTask = {
    labels.foreach(label => {
      val data = mergeData(spark, label)





    })
    this
  }





}
