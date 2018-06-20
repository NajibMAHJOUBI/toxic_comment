package fr.toxic.spark

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
  }

  @Test def testTfIdfTest(): Unit = {
    val data = new LoadDataSetTask("src/test/resources/data").run(spark, "tfIdf")
    val vocabSize = 10
    val tfIdf = new TfIdfTask().run(data)
    tfIdf.write.parquet("src/test/resources/data/labelsFeatures")

    assert(tfIdf.isInstanceOf[DataFrame])
    assert(tfIdf.count() == data.count())
    assert(tfIdf.columns.contains("tf_idf"))
    val idf = tfIdf.rdd.map(x => x.getAs[Vector](x.fieldIndex("tf_idf"))).collect()(0)
    assert(idf.size == vocabSize)
    }

  @After def afterAll() {
    spark.stop()
  }

}
