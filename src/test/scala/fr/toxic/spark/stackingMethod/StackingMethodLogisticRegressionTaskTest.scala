package fr.toxic.spark.stackingMethod

import fr.toxic.spark.classification.stackingMethod.StackingMethodLogisticRegressionTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession
import org.junit.{After, Before, Test}


class StackingMethodLogisticRegressionTaskTest {

  private var spark: SparkSession = _
  private val pathLabel = "/home/mahjoubi/Documents/github/toxic_comment/src/test/resources/data"
  private val pathPrediction = "/home/mahjoubi/Documents/github/toxic_comment/src/test/resources/data/binaryRelevance"
  private val pathSave = "/home/mahjoubi/Documents/github/toxic_comment/target/model/stackingModel/logisticRegression"

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test stacking method test")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testStackingMethodLogisticRegressionTask(): Unit = {
    val methodClassification = Array("logisticRegression", "randomForest")
    val labels = Array("toxic", "obscene")
    val label = "toxic"
    val stackingMethod = new StackingMethodLogisticRegressionTask(labels, methodClassification, pathLabel, pathPrediction, pathSave)
    stackingMethod.run(spark)

    labels.foreach(label => new java.io.File(s"$pathSave/model/$label").exists)
  }

  @After def afterAll() {
    spark.stop()
  }
}
