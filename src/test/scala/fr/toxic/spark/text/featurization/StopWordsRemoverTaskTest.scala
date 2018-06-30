package fr.toxic.spark.text.featurization

import fr.toxic.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
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
    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testStopWordsRemover(): Unit = {
    val data = new LoadDataSetTask(sourcePath = "src/test/resources/data").run(spark, "tokenizer")
    val removed = new StopWordsRemoverTask().run(data)
    // removed.write.parquet("src/test/resources/data/stopWordsRemover")

    assert(removed.isInstanceOf[DataFrame])
    assert(removed.count() == data.count())
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
