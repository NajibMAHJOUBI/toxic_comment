package fr.toxic.spark.kaggle

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession

object KaggleSubmisionStackingMethodExample {

  def main(arguments: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local")
      .appName("Kaggle Submission Example - Stacking Method")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    val multiLabelClassificationMethod: String = "binaryRelevance"
    val validationMethod: String = "cross_validation"
    val pathRoot: String = s"submission/$multiLabelClassificationMethod/$validationMethod"
    val pathLabel: String = ""
    val pathPrediction: String = ""
    val pathSave: String = ""
  }

}
