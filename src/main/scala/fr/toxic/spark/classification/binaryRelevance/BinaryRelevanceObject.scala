package fr.toxic.spark.classification.binaryRelevance

import org.apache.spark.ml.linalg.Vector

object BinaryRelevanceObject {

  def getPredictionProbability(probability: Vector, prediction: Double): Double = {
    probability(prediction.toInt)
  }

}
