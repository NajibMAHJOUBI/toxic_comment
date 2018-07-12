
package fr.toxic.spark.kaggle

import fr.toxic.spark.classification.multiLabelClassification.binaryRelevance.{BinaryRelevanceDecisionTreeTask, BinaryRelevanceGbtClassifierTask, BinaryRelevanceRandomForestTask}
import fr.toxic.spark.classification.task.{DecisionTreeTask, LinearSvcTask, LogisticRegressionTask, RandomForestTask}
import fr.toxic.spark.classification.task.binaryRelevance.{BinaryRelevanceLinearSvcTask, BinaryRelevanceLogisticRegressionTask}
import fr.toxic.spark.text.featurization.{CountVectorizerTask, StopWordsRemoverTask, TfIdfTask, TokenizerTask}
import fr.toxic.spark.utils.{LoadDataSetTask, WriteKaggleSubmission}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}


object KaggleSubmissionBinaryRelevanceExample {

  def main(arguments: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local")
      .appName("Kaggle Submission Example - Binary Relevance")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    val classifierMethods = Array("logistic_regression", "linear_svc", "decision_tree", "random_forest", "gbt_classifier")
    val methodValidation = "cross_validation"
    val labels = Array("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
    val rootPath = s"target/kaggle/binaryRelevance"

    // Train
    val train = new LoadDataSetTask(sourcePath = "data/parquet").run(spark, "train")
    val trainTokens = new TokenizerTask().run(train)
    val trainStopWordsRemoved = new StopWordsRemoverTask(stopWordsOption = "spark").run(trainTokens)
    val countVectorizerModel = new CountVectorizerTask(minDF = 5, vocabSize = 1000)
    countVectorizerModel.run(trainStopWordsRemoved)
    val tfIdfModel = new TfIdfTask()
    tfIdfModel.run(countVectorizerModel.getTransform())
    val trainTfIdf = tfIdfModel.getTransform()

    // Test
    val test = new LoadDataSetTask(sourcePath = "data/parquet").run(spark, "test")
    val testTokens = new TokenizerTask().run(test)
    val testStopWordsRemoved = new StopWordsRemoverTask(stopWordsOption = "spark").run(testTokens)
    val testTf = countVectorizerModel.transform(testStopWordsRemoved).getTransform()
    val testTfIdf = tfIdfModel.transform(testTf).getTransform()

    classifierMethods.foreach(classifierMethod => {
      if (classifierMethod == "linear_svc") {
        val pathMethod = s"$rootPath/$methodValidation/$classifierMethod"
        val binaryRelevance = new BinaryRelevanceLinearSvcTask(columns = labels, savePath = pathMethod,
          featureColumn = "tf_idf", methodValidation = methodValidation)
        binaryRelevance.run(trainTfIdf)
        var prediction: DataFrame = testTfIdf
        labels.foreach(label => {
              binaryRelevance.loadModel(s"$pathMethod/model/$label")
              binaryRelevance.computePrediction(prediction)
              prediction = binaryRelevance.getPrediction})
        new WriteKaggleSubmission().run(prediction, pathMethod)
      } else if (classifierMethod == "decision_tree"){
        val pathMethod = s"$rootPath/$methodValidation/$classifierMethod"
        val binaryRelevance = new BinaryRelevanceDecisionTreeTask(columns = labels, savePath = pathMethod,
          featureColumn = "tf_idf", methodValidation = methodValidation)
        binaryRelevance.run(trainTfIdf)
        var prediction: DataFrame = testTfIdf
        labels.foreach(label => {
              binaryRelevance.loadModel(s"$pathMethod/model/$label")
              binaryRelevance.computePrediction(prediction)
              prediction = binaryRelevance.getPrediction})
        new WriteKaggleSubmission().run(prediction, pathMethod)
      } else if (classifierMethod == "random_forest"){
        val pathMethod = s"$rootPath/$methodValidation/$classifierMethod"
        val binaryRelevance = new BinaryRelevanceRandomForestTask(columns = labels, savePath = pathMethod,
          featureColumn = "tf_idf", methodValidation = methodValidation)
        binaryRelevance.run(trainTfIdf)
        var prediction: DataFrame = testTfIdf
        labels.foreach(label => {
              binaryRelevance.loadModel(s"$pathMethod/model/$label")
              binaryRelevance.computePrediction(prediction)
              prediction = binaryRelevance.getPrediction})
        new WriteKaggleSubmission().run(prediction, pathMethod)
      } else if (classifierMethod == "gbt_classifier"){
        val pathMethod = s"$rootPath/$methodValidation/$classifierMethod"
        val binaryRelevance = new BinaryRelevanceGbtClassifierTask(columns = labels, savePath = pathMethod,
          featureColumn = "tf_idf", methodValidation = methodValidation)
        binaryRelevance.run(trainTfIdf)
        var prediction: DataFrame = testTfIdf
        labels.foreach(label => {
              binaryRelevance.loadModel(s"$pathMethod/model/$label")
              binaryRelevance.computePrediction(prediction)
              prediction = binaryRelevance.getPrediction})
        new WriteKaggleSubmission().run(prediction, pathMethod)
      } else {
        val pathMethod = s"$rootPath/$methodValidation/$classifierMethod"
        val binaryRelevance = new BinaryRelevanceLogisticRegressionTask(columns = labels, savePath = pathMethod,
          featureColumn = "tf_idf", methodValidation = methodValidation)
        binaryRelevance.run(trainTfIdf)
        var prediction: DataFrame = testTfIdf
        labels.foreach(label => {
              binaryRelevance.loadModel(s"$pathMethod/model/$label")
              binaryRelevance.computePrediction(prediction)
              prediction = binaryRelevance.getPrediction})
        new WriteKaggleSubmission().run(prediction, pathMethod)
      }
    }
    )
    }
}

//kaggle competitions submit -c jigsaw-toxic-comment-classification-challenge -f part-00000-bdff9beb-52cf-4ff7-bf26-de79cbf100fc-c000.csv -m "simple validation + logistic regression"