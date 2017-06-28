/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.factorgraph

import cc.factorie.DenseTensor1
import cc.factorie.model._

object FactorUtils {
  def getPairConjunctionFactor(leftVariable: (BinaryRandomVariable, Boolean), rightVariable: (BinaryRandomVariable, Boolean), outputVariable: Option[(BinaryRandomVariable, Boolean)] = None): Factor = {
    if (outputVariable.isEmpty) {
      (leftVariable._2, rightVariable._2) match {
        case (true, true) => PairConjunctionUtils.family2PP.Factor(leftVariable._1, rightVariable._1)
        case (true, false) => PairConjunctionUtils.family2PN.Factor(leftVariable._1, rightVariable._1)
        case (false, true) => PairConjunctionUtils.family2NP.Factor(leftVariable._1, rightVariable._1)
        case (false, false) => PairConjunctionUtils.family2NN.Factor(leftVariable._1, rightVariable._1)
      }
    } else {
      (leftVariable._2, rightVariable._2, outputVariable.get._2) match {
        case (true, true, true) => PairConjunctionUtils.family2WithOutputPPP.Factor(leftVariable._1, rightVariable._1, outputVariable.get._1)
        case (true, true, false) => PairConjunctionUtils.family2WithOutputPPN.Factor(leftVariable._1, rightVariable._1, outputVariable.get._1)
        case (true, false, true) => PairConjunctionUtils.family2WithOutputPNP.Factor(leftVariable._1, rightVariable._1, outputVariable.get._1)
        case (true, false, false) => PairConjunctionUtils.family2WithOutputPNN.Factor(leftVariable._1, rightVariable._1, outputVariable.get._1)
        case (false, true, true) => PairConjunctionUtils.family2WithOutputNPP.Factor(leftVariable._1, rightVariable._1, outputVariable.get._1)
        case (false, true, false) => PairConjunctionUtils.family2WithOutputNPN.Factor(leftVariable._1, rightVariable._1, outputVariable.get._1)
        case (false, false, true) => PairConjunctionUtils.family2WithOutputNNP.Factor(leftVariable._1, rightVariable._1, outputVariable.get._1)
        case (false, false, false) => PairConjunctionUtils.family2WithOutputNNN.Factor(leftVariable._1, rightVariable._1, outputVariable.get._1)
      }
    }
  }

  def getPairDisjunctionFactor(leftVariable: (BinaryRandomVariable, Boolean), rightVariable: (BinaryRandomVariable, Boolean), outputVariable: Option[(BinaryRandomVariable, Boolean)] = None): Factor = {
    if (outputVariable.isEmpty) {
      (leftVariable._2, rightVariable._2) match {
        case (true, true) => PairDisjunctionUtils.family2PP.Factor(leftVariable._1, rightVariable._1)
        case (true, false) => PairDisjunctionUtils.family2PN.Factor(leftVariable._1, rightVariable._1)
        case (false, true) => PairDisjunctionUtils.family2NP.Factor(leftVariable._1, rightVariable._1)
        case (false, false) => PairDisjunctionUtils.family2NN.Factor(leftVariable._1, rightVariable._1)
      }
    } else {
      (leftVariable._2, rightVariable._2, outputVariable.get._2) match {
        case (true, true, true) => PairDisjunctionUtils.family2WithOutputPPP.Factor(leftVariable._1, rightVariable._1, outputVariable.get._1)
        case (true, true, false) => PairDisjunctionUtils.family2WithOutputPPN.Factor(leftVariable._1, rightVariable._1, outputVariable.get._1)
        case (true, false, true) => PairDisjunctionUtils.family2WithOutputPNP.Factor(leftVariable._1, rightVariable._1, outputVariable.get._1)
        case (true, false, false) => PairDisjunctionUtils.family2WithOutputPNN.Factor(leftVariable._1, rightVariable._1, outputVariable.get._1)
        case (false, true, true) => PairDisjunctionUtils.family2WithOutputNPP.Factor(leftVariable._1, rightVariable._1, outputVariable.get._1)
        case (false, true, false) => PairDisjunctionUtils.family2WithOutputNPN.Factor(leftVariable._1, rightVariable._1, outputVariable.get._1)
        case (false, false, true) => PairDisjunctionUtils.family2WithOutputNNP.Factor(leftVariable._1, rightVariable._1, outputVariable.get._1)
        case (false, false, false) => PairDisjunctionUtils.family2WithOutputNNN.Factor(leftVariable._1, rightVariable._1, outputVariable.get._1)
      }
    }
  }

  def getUnaryFactor(variableWithState: (BinaryRandomVariable, Boolean)): Factor = {
    val family = new DotTemplateWithStatistics1[BinaryRandomVariable] with Parameters {
      val weights = Weights(new DenseTensor1(2))
    }

    if (variableWithState._2) {
      family.weights.value(1) = 1.0
      family.weights.value(0) = -1.0
    } else {
      family.weights.value(1) = -1.0
      family.weights.value(0) = 1.0
    }

    family.Factor(variableWithState._1)
  }
}
