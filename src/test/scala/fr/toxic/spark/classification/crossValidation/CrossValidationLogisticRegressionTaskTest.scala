package fr.toxic.spark.classification.crossValidation

import fr.toxic.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
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


class CrossValidationLogisticRegressionTaskTest extends AssertionsForJUnit {

  private val label = "toxic"
  private val feature = "tf"
  private val prediction = s"prediction_${label}"
  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()
    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
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

    val bestModel = cv.getBestModel()
    assert(bestModel.getLabelCol == label)
    assert(bestModel.getFeaturesCol == feature)
    assert(bestModel.getPredictionCol == prediction)

    val transform = cv.transform(data)
    transform.printSchema()
    assert(transform.isInstanceOf[DataFrame])
    assert(transform.count() == data.count())
    assert(transform.columns.contains(prediction))
    }

  @After def afterAll() {
    spark.stop()
  }

}
