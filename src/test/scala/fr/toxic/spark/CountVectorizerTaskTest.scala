package fr.toxic.spark

import fr.toxic.spark.text.featurization.CountVectorizerTask
import fr.toxic.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
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
    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testCountVectorizer(): Unit = {
    val data = new LoadDataSetTask("src/test/resources/data").run(spark, "stopWordsRemover")
    val vocabSize = 10
    val countVectorizer = new CountVectorizerTask(minDF = 1, vocabSize = vocabSize)
    countVectorizer.run(data)
    // count.write.parquet("src/test/resources/data/countVectorizer")

    val transform = countVectorizer.getTransform()
    assert(transform.isInstanceOf[DataFrame])
    assert(transform.count() == data.count())
    assert(transform.columns.contains("tf"))
    val tf = transform.rdd.map(x => x.getAs[Vector](x.fieldIndex("tf"))).collect()(0)
    assert(tf.size == vocabSize)
  }

  @After def afterAll() {
    spark.stop()
  }

}
