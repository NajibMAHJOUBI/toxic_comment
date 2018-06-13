package fr.toxic.spark

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}

/**
  * Created by mahjoubi on 12/06/18.
  */
class LabelFeaturesTask(val featureColumn: String = "features", val tfIdf: String = "tf_idf") {

  def run(data: DataFrame, labelColumn: String): DataFrame = {
    val label = defineLabel(data, labelColumn)
    defineFeatures(label, featureColumn, tfIdf)
  }

  def defineLabel(data: DataFrame, labelColumn: String) = {
    val createLabels = udf((toxic: Float, severe_toxic: Float, obscene: Float, threat: Float, insult: Float, identity_hate: Float) =>
      Vectors.dense(toxic, severe_toxic, obscene, threat, insult, identity_hate))
    data.withColumn(labelColumn, createLabels(col("toxic"), col("severe_toxic"), col("obscene"), col("threat"),
      col("insult"), col("identity_hate")))
  }

  def defineFeatures(data: DataFrame, featureColumn: String, tfIdf: String) = {
    data.withColumnRenamed(tfIdf, featureColumn)
  }

}
