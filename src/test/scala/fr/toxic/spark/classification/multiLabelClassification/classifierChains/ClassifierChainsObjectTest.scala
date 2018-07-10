package fr.toxic.spark.classification.multiLabelClassification.classifierChains

import fr.toxic.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  *
  * Test for classifier chains object (udf)
  *
  */
class ClassifierChainsObjectTest extends AssertionsForJUnit  {

  private var spark: SparkSession = _
  private val size: Int = 3
  private val indices: Array[Int] = Array(0, 2)
  private val values: Array[Double] = Array(1.0, 2.3)
  private val vector = Vectors.sparse(size, indices, values)
  private val value: Long = 10

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test classifier chains")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testGetSize(): Unit = {
    assert(ClassifierChainsObject.getSize(vector) == size)
  }

  @Test def testGetNewIndices(): Unit = {
    val newIndices = ClassifierChainsObject.getNewIndices(vector, size)
    assert(newIndices.length == vector.toSparse.indices.length+1)
    (0 to indices.length-1).foreach(ind => assert(newIndices(ind) == vector.toSparse.indices(ind)))
    assert(newIndices(newIndices.length-1) == size)
  }

  @Test def testGetNewValues(): Unit = {
    val newValues = ClassifierChainsObject.getNewValues(vector, value)
    (0 to values.length-1).foreach(ind => assert(newValues(ind) == vector.toSparse.values(ind)))
    assert(newValues(newValues.length-1) == value)
  }

  @Test def testGetSparseVector(): Unit = {
    val newVector = ClassifierChainsObject.getSparseVector(size, indices, values)
    assert(newVector.equals(vector))
  }

  @Test def testCreateNewFeatures(): Unit = {
    val newVector = ClassifierChainsObject.createNewFeatures(vector, value)
    assert(newVector.size == vector.size + 1)
    // tests on values
    (0 to vector.size - 1).foreach(ind => assert(newVector(ind) == vector(ind)))
    assert(newVector(vector.size) == value)
    // test on indices
    (0 to vector.toSparse.indices.length - 1).foreach(ind => assert(newVector.toSparse.indices(ind) == vector.toSparse.indices(ind)))
    assert(newVector.toSparse.indices(newVector.toSparse.indices.size-1) == vector.size)

    print(vector, newVector)
    println(vector.toSparse.indices.length, newVector.toSparse.indices.length)
    print(vector.getClass, newVector.getClass)
  }

  @After def afterAll() {
    spark.stop()
  }
}
