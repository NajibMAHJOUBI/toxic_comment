package fr.toxic.spark.stackingMethod

import fr.toxic.spark.classification.stackingMethod.{StackingMethodObject, StackingMethodTask}
import fr.toxic.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.junit.{After, Before, Test}


class StackingMethodTaskTest {

  private var spark: SparkSession = _
  private val pathLabel = "/home/mahjoubi/Documents/github/toxic_comment/src/test/resources/data"
  private val pathPrediction = "/home/mahjoubi/Documents/github/toxic_comment/src/test/resources/data/binaryRelevance"

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test stacking method test")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testLoadDataPredictionByLabel(): Unit = {
    val stackingMethod = new StackingMethodTask(Array(""), Array(""), "", pathPrediction)
    val label = "toxic"
    val method = "logisticRegression"
    val data = stackingMethod.loadDataPredictionByLabel(spark, "logisticRegression", label)
    assert(data.isInstanceOf[DataFrame])
    assert(data.columns.length == 2)
    assert(data.columns.contains("id"))
    assert(data.columns.contains(s"prediction_$method"))
  }

  @Test def testLoadDataLabel(): Unit = {
    val stackingMethod = new StackingMethodTask(Array(""), Array(""), pathLabel, "")
    val label = "toxic"
    val data  = stackingMethod.loadDataLabel(spark, label)

    assert(data.isInstanceOf[DataFrame])
    assert(data.columns.length == 2)
    assert(data.columns.contains("id"))
    assert(data.columns.contains("label"))
  }


  @Test def testMergeData(): Unit = {
    val methodClassification = Array("logisticRegression", "randomForest")
    val stackingMethod = new StackingMethodTask(Array(""), methodClassification, pathLabel, pathPrediction)
    stackingMethod.mergeData(spark, "toxic")
    val data = stackingMethod.getData

    assert(data.isInstanceOf[DataFrame])
    assert(data.columns.length == 3)
    assert(data.columns.contains("label"))
    methodClassification.foreach(method => assert(data.columns.contains(s"prediction_$method")))
    }

  @Test def testCreateDfLabelFeatures(): Unit = {
    val methodClassification = Array("logisticRegression", "randomForest")
    val stackingMethod = new StackingMethodTask(Array(""), methodClassification, pathLabel, pathPrediction)
    stackingMethod.mergeData(spark, "toxic")
    val labelFeatures = stackingMethod.createDfLabelFeatures(spark, stackingMethod.getData)

    assert(labelFeatures.isInstanceOf[DataFrame])
    assert(labelFeatures.columns.length == 2)
    assert(labelFeatures.columns.contains("label"))
    assert(labelFeatures.columns.contains("features"))
  }

  @After def afterAll() {
    spark.stop()
  }
}
