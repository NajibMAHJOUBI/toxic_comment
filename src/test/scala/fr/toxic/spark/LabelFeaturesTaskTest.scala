package fr.toxic.spark

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class LabelFeaturesTaskTest extends AssertionsForJUnit  {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()
  }

  @Test def testLabelsFeatures(): Unit = {
    val newLabelColumn="label_toxic"
    val newFeatureColumn="features"
    val data = new LoadDataSetTask("src/test/resources/data").run(spark, "tfIdf")
    val labelFeatures = new LabelFeaturesTask(oldLabelColumn = "toxic", newLabelColumn = newLabelColumn,
                                              oldFeatureColumn = "tf_idf", newFeatureColumn = newFeatureColumn).run(data)

    assert(labelFeatures.isInstanceOf[DataFrame])
    assert(labelFeatures.count() == data.count())
    assert(labelFeatures.columns.contains(newLabelColumn))
    assert(labelFeatures.columns.contains(newFeatureColumn))
    labelFeatures.select(newLabelColumn, newFeatureColumn).show()
    val labels = labelFeatures.rdd.map(x => x.getAs[Vector](x.fieldIndex(newLabelColumn)))//.collect()(0)
    print(labels.collect())
//    assert(labels.size == 6)
//    val features = labelFeatures.rdd.map(x => x.getAs[Vector](x.fieldIndex(newFeatureColumn))).collect()(0)
//    assert(features.size == 10)
  }

  @After def afterAll() {
    spark.stop()
  }
}
