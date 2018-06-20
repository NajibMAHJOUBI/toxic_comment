package fr.toxic.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{LongType, StringType}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 11/06/18.
  */

class LoadDataTaskTest extends AssertionsForJUnit {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()
  }

  @Test def testLoadTrainSet(): Unit = {
    val data = new LoadDataSetTask(sourcePath = "src/test/resources/data").run(spark, "train")
    assert(data.count() == 6)
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
    assert(dataSchema.fields(dataSchema.fieldIndex("toxic")).dataType == LongType)
    assert(dataSchema.fields(dataSchema.fieldIndex("severe_toxic")).dataType == LongType)
    assert(dataSchema.fields(dataSchema.fieldIndex("obscene")).dataType == LongType)
    assert(dataSchema.fields(dataSchema.fieldIndex("threat")).dataType == LongType)
    assert(dataSchema.fields(dataSchema.fieldIndex("insult")).dataType == LongType)
    assert(dataSchema.fields(dataSchema.fieldIndex("identity_hate")).dataType ==  LongType)
  }

  @Test def testLoadTestSet(): Unit = {
    val data = new LoadDataSetTask(sourcePath = "src/test/resources/data").run(spark, "test")
    assert(data.count() == 6)
    val columns = data.columns
    assert(columns.contains("id"))
    assert(columns.contains("comment_text"))
    val dataSchema = data.schema
    assert(dataSchema.fields(dataSchema.fieldIndex("id")).dataType == StringType)
    assert(dataSchema.fields(dataSchema.fieldIndex("comment_text")).dataType == StringType)
  }

  @After def afterAll() {
    spark.stop()
  }

}
