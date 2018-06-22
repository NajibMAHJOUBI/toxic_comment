package fr.toxic.spark

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
    val data = new LoadDataSetTask("src/test/resources/data").run(spark, "tfIdf")
    val logisticRegression = new LogisticRegressionTask(labelColumn = "toxic",
                                                        featureColumn = "tf_idf",
                                                        predictionColumn = "prediction")
    logisticRegression.defineModel()
    logisticRegression.fit(data)
    logisticRegression.transform(data)
    val prediction = logisticRegression.getPrediction()

    assert(prediction.isInstanceOf[DataFrame])
    assert(prediction.columns.contains("prediction"))
    assert(prediction.columns.contains("probability"))
    assert(prediction.columns.contains("rawPrediction"))
  }

  @Test def testRegParam(): Unit = {
    val regParam = 0.5
    val logisticRegression = new LogisticRegressionTask()
    logisticRegression.defineModel()
    logisticRegression.setRegParam(regParam)
    assert(logisticRegression.getRegParam() == regParam)
  }

  @After def afterAll() {
    spark.stop()
  }
}
