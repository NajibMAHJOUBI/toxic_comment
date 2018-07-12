package fr.toxic.spark.utils

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit


class WriteKaggleSubmissionTest extends AssertionsForJUnit  {

  private var spark: SparkSession = _
  private val savePath = "target/model/writeKaggleSubmission"

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()
    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testWriteKaggleSubmission(): Unit = {
    val data = new LoadDataSetTask("src/test/resources/data/binaryRelevance/logisticRegression", format = "csv")
      .run(spark, "prediction")
    val writeKaggle = new WriteKaggleSubmission()
    writeKaggle.run(data, savePath)

    val submission = spark.read.option("header", "true").csv(s"$savePath/submission")
    assert(submission.isInstanceOf[DataFrame])
    assert(submission.count() == data.count())
    assert(submission.columns.length == writeKaggle.getPredictionsColumns().length + 1)
    assert(submission.columns.contains("id"))
    assert(submission.columns.contains("toxic"))
    assert(submission.columns.contains("severe_toxic"))
  }

  @After def afterAll() {
    spark.stop()
  }
}
