package fr.toxic.spark.classification.stackingMethod

import org.apache.spark.sql.Row

object StackingMethodObject {

  def extractValues(p: Row, classificationMethods: Array[String]): Array[Double] = {
    var values: Array[Double] = Array()
    classificationMethods.foreach(method => values = values :+ p.getString(p.fieldIndex(s"prediction_$method")).toDouble)
    values
  }
}