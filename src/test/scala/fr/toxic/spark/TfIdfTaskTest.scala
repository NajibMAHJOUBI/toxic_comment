package fr.toxic.spark

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.linalg.Vector
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
    val data = new LoadDataSetTask("/home/mahjoubi/Documents/github/toxic_comment/src/test/ressources/data")
      .run(spark, "train")
    val tokens = new TokenizerTask().run(data)
    val removed = new StopWordsRemoverTask().run(tokens)
    val vocabSize = 10
    val tf = new CountVectorizerTask(minDF = 1, vocabSize = vocabSize).run(removed)
    val tfIdf = new TfIdfTask().run(tf)

    assert(tfIdf.isInstanceOf[DataFrame])
    assert(tfIdf.columns.contains("tf"))
    val idf = tfIdf.rdd.map(x => x.getAs[Vector](x.fieldIndex("tf"))).collect()(0)
    assert(idf.size == vocabSize)
    }

  @After def afterAll() {
    spark.stop()
  }

}
