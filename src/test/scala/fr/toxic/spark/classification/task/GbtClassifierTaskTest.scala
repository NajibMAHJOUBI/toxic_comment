package fr.toxic.spark.classification.task

import fr.toxic.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class GbtClassifierTaskTest extends AssertionsForJUnit  {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("gbt classifier test")
      .getOrCreate()
    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testGbtClassifier(): Unit = {
    val data = new LoadDataSetTask("src/test/resources/data").run(spark, "tfIdf")
    val gbtClassifier = new GbtClassifierTask(labelColumn = "toxic",
                                             featureColumn = "tf_idf",
                                             predictionColumn = "prediction")
    gbtClassifier.defineModel()
    gbtClassifier.fit(data)
    gbtClassifier.transform(data)
    val transform = gbtClassifier.getTransform

    assert(transform.isInstanceOf[DataFrame])
    assert(transform.columns.contains("prediction"))
    assert(transform.columns.contains("probability"))
    assert(transform.columns.contains("rawPrediction"))
  }

  @Test def testMaxDepth(): Unit = {
    val maxDepth = 5
    val gbtClassifier = new GbtClassifierTask()
    gbtClassifier.defineModel()
    gbtClassifier.setMaxDepth(maxDepth)
    assert(gbtClassifier.getMaxDepth == maxDepth)
  }

  @After def afterAll() {
    spark.stop()
  }
}
