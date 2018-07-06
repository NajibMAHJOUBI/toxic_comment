package fr.toxic.spark.classification.binaryRelevance

import fr.toxic.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class BinaryRelevanceGbtClassifierTaskTest extends AssertionsForJUnit {

  private var spark: SparkSession = _
  private var data: DataFrame = _
  private val savePath: String = "target/model/binaryRelevance"
  private val sourcePath: String = "src/test/resources/data"

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    data = new LoadDataSetTask(sourcePath).run(spark, "tfIdf")
  }

  @Test def testBrOneColumnSimpleValidationTest(): Unit = {
    val columns = Array("toxic")
    new BinaryRelevanceGbtClassifierTask(data = data, columns = columns,
                                         savePath = s"$savePath/oneColumn/simpleValidation",
                                         featureColumn = "tf_idf", methodValidation = "simple").run()
    val prediction = spark.read.option("header", "true").csv(s"$savePath/oneColumn/simpleValidation/prediction")
    assert(prediction.isInstanceOf[DataFrame])
    columns.map(column => assert(prediction.columns.contains(s"label_$column")))
    columns.map(column => assert(prediction.columns.contains(s"prediction_$column")))
    }

  @Test def testBrOneColumnCrossValidationTest(): Unit = {
    val columns = Array("toxic")
    new BinaryRelevanceDecisionTreeTask(columns = columns,
                                        savePath = s"$savePath/oneColumn/crossValidation",
                                        featureColumn = "tf_idf",
                                        methodValidation = "cross_validation").run(data)
    val prediction = spark.read.option("header", "true").csv(s"$savePath/oneColumn/crossValidation/prediction")
    assert(prediction.isInstanceOf[DataFrame])
    columns.map(column => assert(prediction.columns.contains(s"label_$column")))
    columns.map(column => assert(prediction.columns.contains(s"prediction_$column")))
  }

  @Test def testBrTwoColumnSimpleValidationTest(): Unit = {
    val columns = Array("toxic", "severe_toxic")
    new BinaryRelevanceDecisionTreeTask(columns = columns,
                                        savePath = s"$savePath/twoColumn/simpleValidation",
                                        featureColumn = "tf_idf",
                                        methodValidation = "simple").run(data)
    val prediction = spark.read.option("header", "true").csv(s"$savePath/twoColumn/simpleValidation/prediction")
    assert(prediction.isInstanceOf[DataFrame])
    columns.map(column => assert(prediction.columns.contains(s"label_$column")))
    columns.map(column => assert(prediction.columns.contains(s"prediction_$column")))
  }

  @Test def testBrSixColumnSimpleValidationTest(): Unit = {
    val columns =  Array("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
    new BinaryRelevanceDecisionTreeTask(columns = columns,
                                        savePath = s"$savePath/twoColumn/simpleValidation",
                                        featureColumn = "tf_idf",
                                         methodValidation = "simple").run(data)
    val prediction = spark.read.option("header", "true").csv(s"$savePath/twoColumn/simpleValidation/prediction")
    assert(prediction.isInstanceOf[DataFrame])
    columns.map(column => assert(prediction.columns.contains(s"label_$column")))
    columns.map(column => assert(prediction.columns.contains(s"prediction_$column")))
  }


  @After def afterAll() {
    spark.stop()
  }

}
