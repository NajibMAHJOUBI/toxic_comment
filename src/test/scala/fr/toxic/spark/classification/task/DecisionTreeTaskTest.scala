package fr.toxic.spark.classification.task

import fr.toxic.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class DecisionTreeTaskTest extends AssertionsForJUnit  {

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

  @Test def testDecisionTreeClassifier(): Unit = {
    val data = new LoadDataSetTask("src/test/resources/data").run(spark, "tfIdf")
    val decisionTree = new DecisionTreeTask(labelColumn = "toxic",
                                             featureColumn = "tf_idf",
                                             predictionColumn = "prediction")
    decisionTree.defineModel()
    decisionTree.fit(data)
    decisionTree.transform(data)
    val transform = decisionTree.getTransform

    assert(transform.isInstanceOf[DataFrame])
    assert(transform.columns.contains("prediction"))
    assert(transform.columns.contains("probability"))
    assert(transform.columns.contains("rawPrediction"))
  }

  @Test def testMaxDepth(): Unit = {
    val maxDepth = 5
    val decisionTree = new DecisionTreeTask()
    decisionTree.defineModel()
    decisionTree.setMaxDepth(maxDepth)
    assert(decisionTree.getMaxDepth == maxDepth)
  }

  @Test def testMaxBins(): Unit = {
    val maxBins = 5
    val decisionTree = new DecisionTreeTask()
    decisionTree.defineModel()
    decisionTree.setMaxBins(maxBins)
    assert(decisionTree.getMaxDepth == maxBins)
  }

  @After def afterAll() {
    spark.stop()
  }
}
