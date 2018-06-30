package fr.toxic.spark

import fr.toxic.spark.text.featurization.TokenizerTask
import fr.toxic.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.types.{ArrayType, StringType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class TokenizerTaskTest extends AssertionsForJUnit  {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()
    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testTokenizer(): Unit = {
    val data = new LoadDataSetTask("src/test/resources/data").run(spark, "train")
    val tokens = new TokenizerTask().run(data)
    // tokens.write.parquet("src/test/resources/data/tokenizer")

    assert(tokens.isInstanceOf[DataFrame])
    assert(tokens.count() == data.count())
    assert(tokens.columns.contains("words"))
    assert(tokens.schema.fields(tokens.schema.fieldIndex("words")).dataType ==  ArrayType(StringType))
  }

  @After def afterAll() {
    spark.stop()
  }
}
