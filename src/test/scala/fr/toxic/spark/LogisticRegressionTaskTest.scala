package fr.toxic.spark

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class LogisticRegressionTaskTest extends AssertionsForJUnit  {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()
  }

  @Test def testLogisticRegression(): Unit = {
    val data = new LoadDataSetTask("src/test/ressources/data")
      .run(spark, "logisticRegression")
    val logisticRegression = new LogisticRegressionTask()
    logisticRegression.fitModel(data)
    logisticRegression.transformModel(data)
    val prediction = logisticRegression.getPrediction()

    assert(prediction.isInstanceOf[DataFrame])
    assert(prediction.columns.contains("prediction"))
    assert(prediction.columns.contains("probability"))
    assert(prediction.columns.contains("rawPrediction"))
  }

  @After def afterAll() {
    spark.stop()
  }
}
