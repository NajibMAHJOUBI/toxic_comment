
package fr.toxic.spark

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession

object BinaryRelevanceLogisticRegressionExample {


  def main(arguments: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local")
      .appName("BinaryRelevanceLogisticRegressionExample")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)


    val data = new LoadDataSetTask(sourcePath = "data/parquet").run(spark, "train")
    val tokens = new TokenizerTask().run(data)
    val removed = new StopWordsRemoverTask().run(tokens)
    val countWords = new CountVectorizerTask(minDF = 5, vocabSize = 1000).run(removed)
    val tfIdf = new TfIdfTask().run(countWords)

    val columns = Array("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
    val savePath = "target/kaggle/binaryRelevance/twoColumn/simpleValidation"
    new BinaryRelevanceLogisticRegressionTask(columns = columns, savePath = savePath,
                                              featureColumn = "tf_idf", methodValidation = "simple").run(tfIdf)
  }
}

