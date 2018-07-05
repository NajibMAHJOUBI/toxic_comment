
package fr.toxic.spark.kaggle

import fr.toxic.spark.classification.binaryRelevance.{BinaryRelevanceDecisionTreeTask, BinaryRelevanceRandomForestTask}
import fr.toxic.spark.classification.task.{DecisionTreeTask, LinearSvcTask, LogisticRegressionTask, RandomForestTask}
import fr.toxic.spark.classification.task.binaryRelevance.{BinaryRelevanceLinearSvcTask, BinaryRelevanceLogisticRegressionTask}
import fr.toxic.spark.text.featurization.{CountVectorizerTask, StopWordsRemoverTask, TfIdfTask, TokenizerTask}
import fr.toxic.spark.utils.{LoadDataSetTask, WriteKaggleSubmission}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession


object KaggleSubmissionBinaryRelevanceExample {

  def main(arguments: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local")
      .appName("Kaggle Submission Example")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    val classifierMethod = "logistic_regression"
    val methodValidation = "simple"
    val probabilityPrediction: Boolean = true
    val columns = Array("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
    val savePath = s"target/kaggle/binaryRelevance/$methodValidation/$classifierMethod"

    // Train
    val train = new LoadDataSetTask(sourcePath = "data/parquet").run(spark, "train")
    val trainTokens = new TokenizerTask().run(train)
    val trainStopWordsRemoved = new StopWordsRemoverTask().run(trainTokens)
    val countVectorizerModel = new CountVectorizerTask(minDF = 5, vocabSize = 1000)
    countVectorizerModel.run(trainStopWordsRemoved)

    val tfIdfModel = new TfIdfTask()
    tfIdfModel.run(countVectorizerModel.getTransform())

    val trainTfIdf = tfIdfModel.getTransform()

    if (classifierMethod == "linear_svc") {
      val binaryRelevance = new BinaryRelevanceLinearSvcTask(data= trainTfIdf, columns = columns,
                                                             savePath = savePath, featureColumn = "tf_idf",
                                                             methodValidation = methodValidation)
      binaryRelevance.run()
    } else if (classifierMethod == "decision_tree"){
      val binaryRelevance = new BinaryRelevanceDecisionTreeTask(data= trainTfIdf, columns = columns,
                                                                savePath = savePath, featureColumn = "tf_idf",
                                                                methodValidation = methodValidation)
      binaryRelevance.run()
    } else if (classifierMethod == "random_forest"){
      val binaryRelevance = new BinaryRelevanceRandomForestTask(data= trainTfIdf, columns = columns,
                                                                savePath = savePath, featureColumn = "tf_idf",
                                                                methodValidation = methodValidation)
      binaryRelevance.run()
    } else {
      val binaryRelevance = new BinaryRelevanceLogisticRegressionTask(columns = columns,
                                                                      savePath = savePath, featureColumn = "tf_idf",
                                                                      methodValidation = methodValidation,
                                                                      probability = probabilityPrediction)
      binaryRelevance.run(trainTfIdf)
    }


    // Test
    val test = new LoadDataSetTask(sourcePath = "data/parquet").run(spark, "test")
    val testTokens = new TokenizerTask().run(test)
    val testStopWordsRemoved = new StopWordsRemoverTask().run(testTokens)
    val testTf = countVectorizerModel.transform(testStopWordsRemoved).getTransform()
    var testTfIdf = tfIdfModel.transform(testTf).getTransform()

//    val logisticRegression = new LogisticRegressionTask(featureColumn = "tf_idf")
//    val linearSvc = new LinearSvcTask(featureColumn = "tf_idf")
//    val decisionTree = new DecisionTreeTask(featureColumn = "tf_idf")
//    val randomForest = new RandomForestTask(featureColumn = "tf_idf")

    for (column <- columns) {
      if (classifierMethod == "linear_svc") {
        // TODO
      } else if (classifierMethod == "decision_tree"){
        // TODO
      } else if (classifierMethod == "random_forest") {
        // TODO
      } else {
        val binaryRelevance = new BinaryRelevanceLogisticRegressionTask(columns = columns,
                                                                        savePath = savePath, featureColumn = "tf_idf",
                                                                        methodValidation = methodValidation,
                                                                        probability = probabilityPrediction)
        binaryRelevance.loadModel(s"$savePath/model/$column")
        testTfIdf = binaryRelevance.computeProbability(testTfIdf, column)
      }
    }

//    columns.map(column => {
//      if (classifierMethod == "linear_svc") {
//        testTfIdf = linearSvc.loadModel(s"$savePath/model/$column")
//                                      .transform(testTfIdf).getTransform.drop(Seq("rawPrediction", "probability"): _*)
//      } else if (classifierMethod == "decision_tree"){
//        testTfIdf = decisionTree.loadModel(s"$savePath/model/$column")
//                                      .transform(testTfIdf).getTransform.drop(Seq("rawPrediction", "probability"): _*)
//      } else if (classifierMethod == "random_forest") {
//        testTfIdf = randomForest.loadModel(s"$savePath/model/$column")
//                                      .transform(testTfIdf).getTransform.drop(Seq("rawPrediction", "probability"): _*)
//      } else {
//        testTfIdf = logisticRegression.loadModel(s"$savePath/model/$column")
//                                      .transform(testTfIdf).getTransform.drop(Seq("rawPrediction", "probability"): _*)
//      }
//    })

    // testTfIdf.show(5)
    new WriteKaggleSubmission().run(testTfIdf, savePath)


  }
}

//kaggle competitions submit -c jigsaw-toxic-comment-classification-challenge -f part-00000-bdff9beb-52cf-4ff7-bf26-de79cbf100fc-c000.csv -m "simple validation + logistic regression"