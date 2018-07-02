package fr.toxic.spark.classification.task

import fr.toxic.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class RandomForestTaskTest extends AssertionsForJUnit  {

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

  @Test def testRandomForest(): Unit = {
    val data = new LoadDataSetTask("src/test/resources/data").run(spark, "tfIdf")
    val randomForest = new RandomForestTask(labelColumn = "toxic",
                                            featureColumn = "tf_idf",
                                            predictionColumn = "prediction")
    randomForest.defineModel()
    randomForest.fit(data)
    randomForest.transform(data)
    val transform = randomForest.getTransform

    assert(transform.isInstanceOf[DataFrame])
    assert(transform.columns.contains("prediction"))
    assert(transform.columns.contains("probability"))
    assert(transform.columns.contains("rawPrediction"))
  }

  @Test def testMaxDepth(): Unit = {
    val maxDepth = 5
    val randomForest = new RandomForestTask()
    randomForest.defineModel()
    randomForest.setMaxDepth(maxDepth)
    assert(randomForest.getMaxDepth == maxDepth)
  }

  @Test def testMaxBins(): Unit = {
    val maxBins = 5
    val randomForest = new RandomForestTask()
    randomForest.defineModel()
    randomForest.setMaxBins(maxBins)
    assert(randomForest.getMaxDepth == maxBins)
  }

  @After def afterAll() {
    spark.stop()
  }
}
