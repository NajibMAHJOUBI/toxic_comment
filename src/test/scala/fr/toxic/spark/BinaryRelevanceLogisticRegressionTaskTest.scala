package fr.toxic.spark

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class BinaryRelevanceLogisticRegressionTaskTest extends AssertionsForJUnit {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()
  }

  @Test def testBrOneColumnTest(): Unit = {
    val data = new LoadDataSetTask("src/test/resources/data").run(spark, "tfIdf")
    val columns = Array("toxic")
    val savePath = "target/model/binaryRelevance/OneColumn"
    new BinaryRelevanceLogisticRegressionTask(columns = columns, savePath = savePath,
                                              featureColumn = "tf_idf", methodValidation = "simple").run(data)
    val prediction = spark.read.option("header", "true").csv(s"${savePath}/prediction")
    assert(prediction.isInstanceOf[DataFrame])
    columns.map(column => assert(prediction.columns.contains(s"label_${column}")))
    columns.map(column => assert(prediction.columns.contains(s"prediction_${column}")))
    }

  @Test def testBrTwoColumnTest(): Unit = {
    val data = new LoadDataSetTask("src/test/resources/data").run(spark, "tfIdf")
    val columns = Array("toxic", "severe_toxic")
    val savePath = "target/model/binaryRelevance/twoColumn"
    new BinaryRelevanceLogisticRegressionTask(columns = columns, savePath = savePath).run(data)
    val prediction = spark.read.option("header", "true").csv(s"${savePath}/prediction")
    assert(prediction.isInstanceOf[DataFrame])
    columns.map(column => assert(prediction.columns.contains(s"prediction_${column}")))
    columns.map(column => assert(prediction.columns.contains(s"prediction_${column}")))
  }

  @After def afterAll() {
    spark.stop()
  }

}
