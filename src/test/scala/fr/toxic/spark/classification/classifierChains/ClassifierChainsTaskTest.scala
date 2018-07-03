package fr.toxic.spark.classification.classifierChains

import fr.toxic.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit
import fr.toxic.spark.classification.classifierChains.ClassifierChainsObject

/**
  * Created by mahjoubi on 12/06/18.
  *
  * Test for classifier chains task
  *
  */
class ClassifierChainsTaskTest extends AssertionsForJUnit  {

  private var spark: SparkSession = _
  private var data: DataFrame = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test classifier chains")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    data = new LoadDataSetTask("src/test/resources/data").run(spark, "tfIdf")
  }

  @Test def testClassifierChainsTask(): Unit = {
    val addedLabelColumns = "toxic"
    val newFeaturesColumn = "features"
    val featureColumn = "tf_idf"
    val classifierChains = new ClassifierChainsTask(addedLabelColumns = addedLabelColumns,
                                                    outputColumn = "",
                                                    featureColumn = featureColumn, newFeaturesColumn = newFeaturesColumn)
    val newData = classifierChains.run(data)

    val firstRowOne = newData.filter(col(addedLabelColumns) === 1L).rdd.first()
    val oldVectorOne = firstRowOne.getAs[Vector](firstRowOne.fieldIndex(featureColumn))
    val newVectorOne = firstRowOne.getAs[Vector](firstRowOne.fieldIndex(newFeaturesColumn))
    assert(newVectorOne.size == oldVectorOne.size+1)
    (0 until oldVectorOne.toSparse.values.length).foreach(ind => assert(newVectorOne.toSparse.values(ind) == oldVectorOne.toSparse.values(ind)))
    assert(newVectorOne(oldVectorOne.size) == firstRowOne.getLong(firstRowOne.fieldIndex(addedLabelColumns)))
    (0 until oldVectorOne.toSparse.indices.length).foreach(ind => assert(newVectorOne.toSparse.indices(ind) == oldVectorOne.toSparse.indices(ind)))
    assert(newVectorOne.toSparse.indices(newVectorOne.toSparse.indices.size-1) == oldVectorOne.size)

    val firstRowZero = newData.filter(col(addedLabelColumns) === 0L).rdd.first()
    val oldVectorZero = firstRowZero.getAs[Vector](firstRowZero.fieldIndex(featureColumn))
    val newVectorZero = firstRowZero.getAs[Vector](firstRowZero.fieldIndex(newFeaturesColumn))
    assert(newVectorZero.size == oldVectorZero.size+1)
    (0 until oldVectorZero.toSparse.values.length).foreach(ind => assert(newVectorZero.toSparse.values(ind) == oldVectorZero.toSparse.values(ind)))
    (0 until oldVectorZero.toSparse.indices.length).foreach(ind => assert(newVectorZero.toSparse.indices(ind) == oldVectorZero.toSparse.indices(ind)))
  }

  @After def afterAll() {
    spark.stop()
  }
}
