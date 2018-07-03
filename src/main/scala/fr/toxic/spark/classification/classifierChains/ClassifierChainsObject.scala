package fr.toxic.spark.classification.classifierChains

import org.apache.spark.ml.linalg.{Vector, Vectors}

object ClassifierChainsObject {

  def getSize(vector: Vector): Int = {
    vector.size
  }

  def getNewIndices(vector: Vector, value: Int): Array[Int] = {
    vector.toSparse.indices :+ value
  }

  def getNewValues(vector: Vector, value: Double): Array[Double] = {
    vector.toSparse.values :+ value
  }

  def getSparseVector(size: Int, indices: Array[Int], values: Array[Double]): Vector = {
    Vectors.sparse(size, indices, values)
  }

  def createNewFeatures (vector: Vector, value: Long): Vector = {
    val size = getSize(vector)
    val newFeatures = if(value == 0L) {
      getSparseVector(size+1, vector.toSparse.indices, vector.toSparse.values)
          } else {
      val size = getSize(vector)
      val indices: Array[Int] = getNewIndices(vector, size) //vector.toSparse.indices :+ size
      val values: Array[Double] = getNewValues(vector, value) // vector.toSparse.values :+ value.toDouble
      getSparseVector(size+1, indices, values)
    }
    newFeatures
  }

}
