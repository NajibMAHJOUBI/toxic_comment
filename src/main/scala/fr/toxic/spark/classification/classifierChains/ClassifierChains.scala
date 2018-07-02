package fr.toxic.spark.classification.classifierChains

import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.ml.linalg.Vector

class ClassifierChains(val newLabelColumns: Array[String],
                       val outputColumn: String,
                       featureColumn: String) {

  def run(data: DataFrame): Unit = {

  }

  def getFeaturesVector(data: DataFrame): Vector ={
    data.rdd.map((x: Row) => x.getAs[Vector](x.fieldIndex(featureColumn))).first()
  }

//  def sizeFeature(u: Vector): Int = {
//    u.size
//  }
//
//  def getIndices(u: Vector): Array[Int] = {
//    u.toSparse.indices
//  }
//
//  def getValues(u: Vector): Array[Double] = {
//    u.toSparse.values
//  }

  def addToIndices(uVector: Vector): Array[Int] = {
    uVector.toSparse.indices :+ (uVector.size + 1)
  }

  def addToValues(uVector: Vector, value: Double): Array[Double] = {
    uVector.toSparse.values :+ value
  }

  def createNewFeatures(uVector)

  def createNewFeatures(data: DataFrame): Unit = {
    val vector = getFeaturesVector(data)
  }

}
