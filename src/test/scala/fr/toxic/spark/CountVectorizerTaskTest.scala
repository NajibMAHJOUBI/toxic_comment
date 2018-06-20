package fr.toxic.spark

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class CountVectorizerTaskTest extends AssertionsForJUnit  {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()
  }

  @Test def testCountVectoizer(): Unit = {
    val data = new LoadDataSetTask("src/test/resources/data").run(spark, "countVectorizer")
    val vocabSize = 10
    val count = new CountVectorizerTask(minDF = 1, vocabSize = vocabSize).run(data)
//    count.write.parquet("src/test/resources/data/tfIdf")

    assert(count.isInstanceOf[DataFrame])
    assert(count.count() == data.count())
    assert(count.columns.contains("tf"))
    val tf = count.rdd.map(x => x.getAs[Vector](x.fieldIndex("tf"))).collect()(0)
    assert(tf.size == vocabSize)
  }

  @After def afterAll() {
    spark.stop()
  }
}
