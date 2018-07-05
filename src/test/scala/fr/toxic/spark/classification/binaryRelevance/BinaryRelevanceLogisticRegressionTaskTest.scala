package fr.toxic.spark.classification.binaryRelevance

import fr.toxic.spark.classification.task.binaryRelevance.BinaryRelevanceLogisticRegressionTask
import fr.toxic.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class BinaryRelevanceLogisticRegressionTaskTest extends AssertionsForJUnit {

  private var spark: SparkSession = _
  private var data: DataFrame = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test BR logistic regression")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    data = new LoadDataSetTask("src/test/resources/data").run(spark, "tfIdf")
  }

  @Test def testBrOneColumnSimpleValidationTest(): Unit = {
    val columns = Array("toxic")
    val savePath = "target/model/binaryRelevance/oneColumn/simpleValidation"
    new BinaryRelevanceLogisticRegressionTask(data = data, columns = columns, savePath = savePath,
                                              featureColumn = "tf_idf", methodValidation = "simple").run()
    val prediction = spark.read.option("header", "true").csv(s"${savePath}/prediction")
    assert(prediction.isInstanceOf[DataFrame])
    columns.map(column => assert(prediction.columns.contains(s"label_${column}")))
    columns.map(column => assert(prediction.columns.contains(s"prediction_${column}")))
    }

  @Test def testBrOneColumnCrossValidationTest(): Unit = {
    val columns = Array("toxic")
    val savePath = "target/model/binaryRelevance/oneColumn/crossValidation"
    new BinaryRelevanceLogisticRegressionTask(data= data, columns = columns, savePath = savePath,
      featureColumn = "tf_idf", methodValidation = "cross_validation").run()
    val prediction = spark.read.option("header", "true").csv(s"${savePath}/prediction")
    assert(prediction.isInstanceOf[DataFrame])
    columns.map(column => assert(prediction.columns.contains(s"label_${column}")))
    columns.map(column => assert(prediction.columns.contains(s"prediction_${column}")))
  }

  @Test def testBrTwoColumnSimpleValidationTest(): Unit = {
    val columns = Array("toxic", "severe_toxic")
    val savePath = "target/model/binaryRelevance/twoColumn/simpleValidation"
    new BinaryRelevanceLogisticRegressionTask(data= data, columns = columns, savePath = savePath,
                                              featureColumn = "tf_idf", methodValidation = "simple").run()
    val prediction = spark.read.option("header", "true").csv(s"${savePath}/prediction")
    assert(prediction.isInstanceOf[DataFrame])
    columns.map(column => assert(prediction.columns.contains(s"prediction_${column}")))
    columns.map(column => assert(prediction.columns.contains(s"prediction_${column}")))
  }

  @Test def testBrSixColumnSimpleValidationTest(): Unit = {
    val columns =  Array("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
    val savePath = "target/model/binaryRelevance/twoColumn/simpleValidation"
    new BinaryRelevanceLogisticRegressionTask(data= data, columns = columns, savePath = savePath,
                                              featureColumn = "tf_idf", methodValidation = "simple").run()
    val prediction = spark.read.option("header", "true").csv(s"${savePath}/prediction")
    assert(prediction.isInstanceOf[DataFrame])
    columns.map(column => assert(prediction.columns.contains(s"prediction_${column}")))
    columns.map(column => assert(prediction.columns.contains(s"prediction_${column}")))
  }


  @After def afterAll() {
    spark.stop()
  }

}
