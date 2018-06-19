package fr.toxic.spark

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class CrossValidationTaskTest extends AssertionsForJUnit {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }

  @Test def crossValidationTest(): Unit = {
    val data = new LoadDataSetTask("src/test/resources/data").run(spark, "tfIdf")
    val label = "toxic"
    val feature = "tf"
    val prediction = "prediction"
    val model = "logistic_regression"
    val cv = new CrossValidationTask(data = data,
      labelColumn = label,
      featureColumn = feature,
      predictionColumn = prediction,
      modelClassifier = model,
      pathModel = "", pathPrediction = "")
    cv.run()

    assert(cv.getLabelColumn() == label)
    assert(cv.getFeatureColumn() == feature)
    assert(cv.getPredictionColumn() == prediction)
    assert(cv.getModelClassifier() == model)
    }

  @After def afterAll() {
    spark.stop()
  }

}
