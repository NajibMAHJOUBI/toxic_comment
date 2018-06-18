package fr.toxic.spark

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class LabelsFeaturesTaskTest extends AssertionsForJUnit  {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()
  }

  @Test def testLabelsFeatures(): Unit = {
    val data = new LoadDataSetTask("src/test/resources/data").run(spark, "train")
    val tokens = new TokenizerTask().run(data)
    val removed = new StopWordsRemoverTask().run(tokens)
    val vocabSize = 10
    val count = new CountVectorizerTask(minDF = 1, vocabSize = vocabSize).run(removed)
    val tfIdf = new TfIdfTask().run(count)
    val labelFeatures = new LabelsFeaturesTask().run(tfIdf)

    assert(labelFeatures.isInstanceOf[DataFrame])
    assert(labelFeatures.columns.contains("label"))
    assert(labelFeatures.columns.contains("features"))
    val labels = labelFeatures.rdd.map(x => x.getAs[Vector](x.fieldIndex("label"))).collect()(0)
    assert(labels.size == 6)
    val features = labelFeatures.rdd.map(x => x.getAs[Vector](x.fieldIndex("features"))).collect()(0)
    assert(features.size == vocabSize)
  }

  @After def afterAll() {
    spark.stop()
  }
}
