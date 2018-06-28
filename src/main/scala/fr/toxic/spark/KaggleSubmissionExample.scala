
package fr.toxic.spark

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession

object KaggleSubmissionExample {

  def main(arguments: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local")
      .appName("Kaggle Submission Example")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    // Train
    val train = new LoadDataSetTask(sourcePath = "data/parquet").run(spark, "train")
    val trainTokens = new TokenizerTask().run(train)
    val trainStopWordsRemoved = new StopWordsRemoverTask().run(trainTokens)
    val countVectorizerModel = new CountVectorizerTask(minDF = 5, vocabSize = 1000)
    countVectorizerModel.run(trainStopWordsRemoved)

    val tfIdfModel = new TfIdfTask()
    tfIdfModel.run(countVectorizerModel.getTransform())

    val trainTfIdf = tfIdfModel.getTransform()

    val columns = Array("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
    val savePath = "target/kaggle/binaryRelevance/twoColumn/simpleValidation"
    val binaryRelevance = new BinaryRelevanceLogisticRegressionTask(data= trainTfIdf, columns = columns, savePath = savePath,
                                              featureColumn = "tf_idf", methodValidation = "simple")
    binaryRelevance.run()
//     new WriteKaggleSubmission().run(binaryRelevance.getPrediction(), savePath)

    // Test
    val test = new LoadDataSetTask(sourcePath = "data/parquet").run(spark, "test")
    val testTokens = new TokenizerTask().run(test)
    val testStopWordsRemoved = new StopWordsRemoverTask().run(testTokens)
    val testTf = countVectorizerModel.transform(testStopWordsRemoved).getTransform()
    var testTfIdf = tfIdfModel.transform(testTf).getTransform()

    val logisticRegression = new LogisticRegressionTask(featureColumn = "tf_idf")
    columns.map(column => {
      testTfIdf = logisticRegression.loadModel(s"$savePath/model/$column").transform(testTfIdf).getTransform()
    })

    //    trainTfIdf.show(5)
    testTfIdf.show(5)

  }
}

//kaggle competitions submit -c jigsaw-toxic-comment-classification-challenge -f part-00000-bdff9beb-52cf-4ff7-bf26-de79cbf100fc-c000.csv -m "simple validation + logistic regression"