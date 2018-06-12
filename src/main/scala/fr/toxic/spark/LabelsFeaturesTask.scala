package fr.toxic.spark

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.col

/**
  * Created by mahjoubi on 12/06/18.
  */
class LabelsFeaturesTask(val labelColumn: String = "label", val featureColumn: String = "features", val tfIdf: String = "tf_idf") {

  def run(data: DataFrame): DataFrame = {
    val label = defineLabls(data, labelColumn)
    defineFeatures(label, featureColumn, tfIdf)
  }

  def defineLabls(data: DataFrame, labelColumn: String) = {
    val createLabels = udf((toxic: Float, severe_toxic: Float, obscene: Float, threat: Float, insult: Float, identity_hate: Float) =>
      Vectors.dense(toxic, severe_toxic, obscene, threat, insult, identity_hate))
    data.withColumn(labelColumn, createLabels(col("toxic"), col("severe_toxic"), col("obscene"), col("threat"),
      col("insult"), col("identity_hate")))
  }

  def defineFeatures(data: DataFrame, featureColumn: String, tfIdf: String) = {
    data.withColumnRenamed(tfIdf, featureColumn)
  }

}
