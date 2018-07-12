package fr.toxic.spark.classification.stackingMethod

import org.apache.spark.sql.Row

object StackingMethodObject {

  def extractVector(p: Row, classificationMethods: Array[String]): Array[Double] = {
    var value: Array[Double] = Array()
    classificationMethods.foreach(method => value = value :+ p.getString(p.fieldIndex(s"prediction_$method")).toDouble)
    value
  }
}