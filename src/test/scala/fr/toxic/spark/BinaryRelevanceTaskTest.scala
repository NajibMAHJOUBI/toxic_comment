package fr.toxic.spark

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class BinaryRelevanceTaskTest extends AssertionsForJUnit {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()
  }

  @Test def testBrOneColumnTest(): Unit = {
    val data = new LoadDataSetTask("/home/mahjoubi/Documents/github/toxic_comment/src/test/ressources/data")
      .run(spark, "train")
    val tokens = new TokenizerTask().run(data)
    val removed = new StopWordsRemoverTask().run(tokens)
    val vocabSize = 10
    val tf = new CountVectorizerTask(minDF = 1, vocabSize = vocabSize).run(removed)
    val tfIdf = new TfIdfTask().run(tf)
    val columns = Array("toxic")
    val savePath = "/home/mahjoubi/Documents/github/toxic_comment/target/model/binaryRelevance/OneColumn"
    new BinaryRelevanceTask(columns = columns, savePath = savePath).run(tfIdf)
    val prediction = spark.read.option("header", "true").csv(s"${savePath}/prediction")
    assert(prediction.isInstanceOf[DataFrame])
    columns.map(column => assert(prediction.columns.contains(s"label_${column}")))
    columns.map(column => assert(prediction.columns.contains(s"prediction_${column}")))
    }

  @Test def testBrTwoColumnTest(): Unit = {
    val data = new LoadDataSetTask("/home/mahjoubi/Documents/github/toxic_comment/src/test/ressources/data")
      .run(spark, "train")
    val tokens = new TokenizerTask().run(data)
    val removed = new StopWordsRemoverTask().run(tokens)
    val vocabSize = 10
    val tf = new CountVectorizerTask(minDF = 1, vocabSize = vocabSize).run(removed)
    val tfIdf = new TfIdfTask().run(tf)
    val columnsToKeep = Array("toxic", "severe_toxic")
    val savePath = "/home/mahjoubi/Documents/github/toxic_comment/target/model/binaryRelevance/twoColumn"
    new BinaryRelevanceTask(columns = columnsToKeep, savePath = savePath).run(tfIdf)

    val prediction = spark.read.csv(savePath)
    assert(prediction.isInstanceOf[DataFrame])
    columnsToKeep.map(column => assert(prediction.columns.contains(s"prediction_${column}")))
  }

  @After def afterAll() {
    spark.stop()
  }

}
