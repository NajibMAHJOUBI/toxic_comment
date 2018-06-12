package fr.toxic.spark

import org.apache.spark.sql.{DataFrame}
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.sql.functions.{col, regexp_replace, udf}

import scala.collection.mutable.WrappedArray

/**
  * Created by mahjoubi on 12/06/18.
  */
class TokenizerTask(val inputColumn: String = "comment_text", val outputColumn: String = "words") {

  def run(data: DataFrame): DataFrame = {
    var tokens: DataFrame = cleanSpecialCharacter(data, inputColumn, "clean_tokens")
    tokens = createTokens(tokens, "clean_tokens", "tokens")
    tokens = removeShortToken(tokens, "tokens", "short_tokens")
    tokens = removeDigit(tokens, "short_tokens", outputColumn)
    tokens
  }

  def createTokens(data: DataFrame, inputColumn: String, outputColumn: String): DataFrame = {
    val tokenizer = new Tokenizer().setInputCol(inputColumn).setOutputCol(outputColumn)
    tokenizer.transform(data)
  }

  def cleanSpecialCharacter(data: DataFrame, inputColumn: String, outputColumn: String): DataFrame = {
    data.withColumn(outputColumn, regexp_replace(data(inputColumn), "[^0-9A-Za-zàâçéèêëîïôûùüÿñæœ]", " "))
  }

  def removeDigit(data: DataFrame, inputColumn: String, outputColumn: String): DataFrame = {
    def udfRemoveDigit = udf((x: WrappedArray[String]) => x.filter(!_.head.isDigit))
    data.withColumn(outputColumn, udfRemoveDigit(data(inputColumn))).drop(inputColumn)
  }

  def removeShortToken(data: DataFrame, inputColumn: String, outputColumn: String): DataFrame = {
    val myUdf = udf((x: WrappedArray[String]) => x.filter(_.length >= 3))
    data.withColumn(outputColumn, myUdf(col(inputColumn))).drop(inputColumn)
  }
}
