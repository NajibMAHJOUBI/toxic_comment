package fr.toxic.spark

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

class WriteKaggleFormat {

  val predictionColumns = Array("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")

  def run(data: DataFrame, savePath: String): Unit = {
    var dataKaggle = data

    predictionColumns.map(column => {
      dataKaggle = dataKaggle.withColumnRenamed(s"prediction_$column", column)
    })

    val columnsToSelect = (Set("id") ++ predictionColumns.toSet).map(name => col(name))

    dataKaggle.select(columnsToSelect.toSeq: _*)
      .write.option("header", "true")
      .mode("overwrite")
      .csv(s"$savePath/submission")
  }

}
