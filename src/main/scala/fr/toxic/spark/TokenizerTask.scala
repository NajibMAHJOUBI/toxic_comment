package fr.toxic.spark

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.Tokenizer

/**
  * Created by mahjoubi on 12/06/18.
  */
class TokenizerTask {

  def run(data: DataFrame): DataFrame = {
    createTokens(data)
  }

  def createTokens(data: DataFrame): DataFrame = {
    val tokenizer = new Tokenizer().setInputCol("comment_text").setOutputCol("words")
    tokenizer.transform(data)
  }
}
