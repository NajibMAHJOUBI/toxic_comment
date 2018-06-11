package fr.toxic.spark

import org.scalatest.junit.AssertionsForJUnit
import org.junit.{After, Before, Test}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StringType

import fr.toxic.spark.LoadDataSet

/**
  * Created by mahjoubi on 11/06/18.
  */

class LoadDataTest extends AssertionsForJUnit {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()
  }

  @Test def testLoadTrainSet(): Unit = {
    val data = new LoadDataSet().run(spark, "train")
    val columns = data.columns
    assert(columns.contains("id"))
    assert(columns.contains("comment_text"))
    assert(columns.contains("toxic"))
    assert(columns.contains("severe_toxic"))
    assert(columns.contains("obscene"))
    assert(columns.contains("threat"))
    assert(columns.contains("insult"))
    assert(columns.contains("identity_hate"))
    val dataSchema = data.schema
    assert(dataSchema.fields(dataSchema.fieldIndex("id")).dataType == StringType)
    assert(dataSchema.fields(dataSchema.fieldIndex("comment_text")).dataType == StringType)
  }

  @After def afterAll() {
    spark.stop()
  }


}
