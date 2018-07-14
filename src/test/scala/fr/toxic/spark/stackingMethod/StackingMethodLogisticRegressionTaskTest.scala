package fr.toxic.spark.stackingMethod

import fr.toxic.spark.classification.stackingMethod.StackingMethodLogisticRegressionTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession
import org.junit.{After, Before, Test}


class StackingMethodLogisticRegressionTaskTest {

  private val pathLabel: String = "src/test/resources/data"
  private val pathPrediction: String = "src/test/resources/data/binaryRelevance"
  private val pathSave: String = "target/model/stackingLogisticRegression"
  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test stacking method test")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testStackingLogisticRegressionMethodTask(): Unit = {
    val methodClassifications = Array("logisticRegression", "randomForest")
    val labels = Array("toxic", "obscene")
    val stackingLogisticRegressionMethod = new StackingMethodLogisticRegressionTask(labels, methodClassifications, pathLabel, pathPrediction, pathSave)
    stackingLogisticRegressionMethod.run(spark)
  }

  @After def afterAll() {
    spark.stop()
  }
}
