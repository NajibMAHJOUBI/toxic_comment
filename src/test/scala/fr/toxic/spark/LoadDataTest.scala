package fr.toxic.spark

import org.scalatest.junit.AssertionsForJUnit
import org.junit.{After, Before, Test}
import org.apache.spark.sql.SparkSession

/**
  * Created by mahjoubi on 11/06/18.
  */

class LoadDataTest extends AssertionsForJUnit {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()
  }

  @Test def testLoadDataSet(): Unit = {
    print("test load dataset")
    val ux = 1.0
    assert(ux == 1.0)
  }

  @After def afterAll() {
    spark.stop()
  }


}
