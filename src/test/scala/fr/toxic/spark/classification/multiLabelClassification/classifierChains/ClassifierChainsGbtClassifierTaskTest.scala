package fr.toxic.spark.classification.multiLabelClassification.classifierChains

import fr.toxic.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  *
  * Test for classifier chains task - linear svc
  *
  */
class ClassifierChainsGbtClassifierTaskTest extends AssertionsForJUnit  {

  private var spark: SparkSession = _
  private var data: DataFrame = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test classifier chains - linear svc classifier")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    data = new LoadDataSetTask("src/test/resources/data").run(spark, "tfIdf")
  }

  @Test def testClassifierChainsSimpleValidationTaskTest(): Unit = {
    val labelColumns: Array[String] = Array("toxic", "severe_toxic")
    val featureColumn = "tf_idf"
    val methodValidation = "simple"
    val savePath = "target/model/classifierChains/simpleValidation/gbtClassifier"
    val classifierChains = new ClassifierChainsGbtClassifierTask(labelColumns= labelColumns,
                                                                featureColumn= featureColumn,
                                                                methodValidation= methodValidation,
                                                                savePath= savePath)
    classifierChains.run(data)
  }

  @Test def testClassifierChainsLrCrossValidationTaskTest(): Unit = {
    val labelColumns: Array[String] = Array("toxic", "severe_toxic", "obscene")
    val featureColumn = "tf_idf"
    val methodValidation = "cross_validation"
    val savePath = "target/model/classifierChains/crossValidation/gbtClassifier"
    val classifierChains = new ClassifierChainsGbtClassifierTask(labelColumns= labelColumns,
                                                                featureColumn= featureColumn,
                                                                methodValidation= methodValidation,
                                                                savePath= savePath)
    classifierChains.run(data)
  }

  @After def afterAll() {
    spark.stop()
  }
}
