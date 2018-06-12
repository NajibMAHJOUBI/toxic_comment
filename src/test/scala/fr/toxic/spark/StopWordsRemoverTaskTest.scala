package fr.toxic.spark

import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.types.{ArrayType, StringType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

import scala.collection.mutable.WrappedArray

/**
  * Created by mahjoubi on 12/06/18.
  */
class StopWordsRemoverTaskTest extends AssertionsForJUnit  {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()
  }

  @Test def testStopWordsRemover(): Unit = {
    val data = new LoadDataSetTask("/home/mahjoubi/Documents/github/toxic_comment/src/test/ressources/data")
      .run(spark, "train")
    val tokens = new TokenizerTask().run(data)
    val removed = new StopWordsRemoverTask().run(tokens)

    assert(removed.isInstanceOf[DataFrame])
    assert(removed.columns.contains("words"))
    assert(removed.schema.fields(removed.schema.fieldIndex("words")).dataType ==  ArrayType(StringType))

    val stopWordsRemover = StopWordsRemover.loadDefaultStopWords("english")
    val words = removed.rdd.map(x => x.getAs[WrappedArray[String]](x.fieldIndex("words"))).collect()
    assert(stopWordsRemover.intersect(words).length == 0)
  }

  @After def afterAll() {
    spark.stop()
  }
}
