package fr.toxic.spark

import fr.toxic.spark.text.featurization.TfIdfTask
import fr.toxic.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class TfIdfTaskTest extends AssertionsForJUnit {

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

  @Test def testTfIdfTest(): Unit = {
    val data = new LoadDataSetTask("src/test/resources/data").run(spark, "countVectorizer")
    val vocabSize = 10
    val tfIdf = new TfIdfTask()
    tfIdf.run(data)
    // tfIdf.write.parquet("src/test/resources/data/tfIdf")

    val transform = tfIdf.getTransform()
    assert(transform.isInstanceOf[DataFrame])
    assert(transform.count() == data.count())
    assert(transform.columns.contains("tf_idf"))
    val idf = transform.rdd.map(x => x.getAs[Vector](x.fieldIndex("tf_idf"))).collect()(0)
    assert(idf.size == vocabSize)
    }

  @After def afterAll() {
    spark.stop()
  }

}
