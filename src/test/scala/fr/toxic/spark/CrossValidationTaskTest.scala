package fr.toxic.spark

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */


class CrossValidationTaskTest extends AssertionsForJUnit {

  private var spark: SparkSession = _
  private val label = "toxic"
  private val feature = "tf"
  private val prediction = "prediction"
  private val model = "logistic_regression"

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

    val cv = new CrossValidationLogisticRegressionTask(data = data,
      labelColumn = label,
      featureColumn = feature,
      predictionColumn = prediction,
      pathModel = "", pathPrediction = "")
    cv.run()

    assert(cv.getLabelColumn() == label)
    assert(cv.getFeatureColumn() == feature)
    assert(cv.getPredictionColumn() == prediction)
    assert(cv.getGridParameters().isInstanceOf[Array[ParamMap]])
    assert(cv.getEstimator().isInstanceOf[LogisticRegression])
    assert(cv.getEvaluator().isInstanceOf[Evaluator])
    assert(cv.getCrossValidator().isInstanceOf[CrossValidator])
    }

  @Test def estimatorEvaluatorTest(): Unit = {
    val data = new LoadDataSetTask("src/test/resources/data").run(spark, "tfIdf")

    val cv = new CrossValidationLogisticRegressionTask(data = data,
      labelColumn = label,
      featureColumn = feature,
      predictionColumn = prediction,
      pathModel = "", pathPrediction = "")
    cv.run()

    val transform = cv.getEstimator().fit(data).transform(data)
    assert(transform.isInstanceOf[DataFrame])
    assert(transform.columns.contains(prediction))

    assert(cv.getEvaluator().evaluate(transform).isInstanceOf[Double])

  }

  @Test def crossValidatorTest(): Unit = {
    val data = new LoadDataSetTask("src/test/resources/data").run(spark, "tfIdf")
    val cv = new CrossValidationLogisticRegressionTask(data = data,
      labelColumn = label,
      featureColumn = feature,
      predictionColumn = prediction,
      pathModel = "", pathPrediction = "")
    cv.run()

    val crossValidator = cv.getCrossValidator()
    crossValidator.fit(data)

  }

  @After def afterAll() {
    spark.stop()
  }

}
