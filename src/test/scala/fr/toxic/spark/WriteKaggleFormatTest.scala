package fr.toxic.spark

import org.apache.spark.sql.types.{ArrayType, StringType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit


class WriteKaggleSubmissionTest extends AssertionsForJUnit  {

  private var spark: SparkSession = _
  private val savePath = "src/test/resources/data"

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()
  }

  @Test def testWriteKaggleSubmission(): Unit = {
    val data = new LoadDataSetTask("src/test/resources/data").run(spark, "binaryRelevance")
    val writeKaggle = new WriteKaggleSubmission()
    writeKaggle.run(data, savePath)

    val submission = spark.read.option("header", "true").csv(s"$savePath/submission")
    submission.show(5)
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
