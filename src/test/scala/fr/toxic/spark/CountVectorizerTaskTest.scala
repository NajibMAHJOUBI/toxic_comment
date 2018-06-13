package fr.toxic.spark

import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.types.{ArrayType, StringType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit
import org.apache.spark.ml.linalg.Vector

import scala.collection.mutable.WrappedArray

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
    val data = new LoadDataSetTask("src/test/ressources/data")
      .run(spark, "train")
    val tokens = new TokenizerTask().run(data)
    val removed = new StopWordsRemoverTask().run(tokens)
    val vocabSize = 10
    val count = new CountVectorizerTask(minDF = 1, vocabSize = vocabSize).run(removed)

    assert(count.isInstanceOf[DataFrame])
    assert(count.columns.contains("tf"))
    val tf = count.rdd.map(x => x.getAs[Vector](x.fieldIndex("tf"))).collect()(0)
    assert(tf.size == vocabSize)
  }

  @After def afterAll() {
    spark.stop()
  }
}
