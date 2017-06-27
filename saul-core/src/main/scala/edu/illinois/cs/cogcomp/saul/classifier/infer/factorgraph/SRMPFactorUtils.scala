/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.factorgraph

object SRMPFactorUtils {
  def getPairConjunctionCosts(leftVariableState: Boolean, rightVariableState: Boolean, outputVariableState: Option[Boolean] = None): Array[Double] = {
    if (outputVariableState.isEmpty) {
      (leftVariableState, rightVariableState) match {
        case (true, true) => Array(0.0, 0.0, 0.0, -1.0)
        case (true, false) => Array(0.0, 0.0, -1.0, 0.0)
        case (false, true) => Array(0.0, -1.0, 0.0, 0.0)
        case (false, false) => Array(-1.0, 0.0, 0.0, 0.0)
      }
    } else {
      (leftVariableState, rightVariableState, outputVariableState.get) match {
        case (true, true, true) => Array(-1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, -1.0)
        case (true, true, false) => Array(0.0, -1.0, 0.0, -1.0, 0.0, -1.0, -1.0, 0.0)
        case (true, false, true) => Array(-1.0, 0.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0)
        case (true, false, false) => Array(0.0, -1.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0)
        case (false, true, true) => Array(-1.0, 0.0, 0.0, -1.0, -1.0, 0.0, -1.0, 0.0)
        case (false, true, false) => Array(0.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, -1.0)
        case (false, false, true) => Array(0.0, -1.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0)
        case (false, false, false) => Array(-1.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0)
      }
    }
  }

  def getPairDisjunctionCosts(leftVariableState: Boolean, rightVariableState: Boolean, outputVariableState: Option[Boolean] = None): Array[Double] = {
    if (outputVariableState.isEmpty) {
      (leftVariableState, rightVariableState) match {
        case (true, true) => Array(0.0, -1.0, -1.0, -1.0)
        case (true, false) => Array(-1.0, 0.0, -1.0, -1.0)
        case (false, true) => Array(-1.0, -1.0, 0.0, -1.0)
        case (false, false) => Array(-1.0, -1.0, -1.0, 0.0)
      }
    } else {
      (leftVariableState, rightVariableState, outputVariableState.get) match {
        case (true, true, true) => Array(-1.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0)
        case (true, true, false) => Array(0.0, -1.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0)
        case (true, false, true) => Array(0.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, -1.0)
        case (true, false, false) => Array(-1.0, 0.0, 0.0, -1.0, -1.0, 0.0, -1.0, 0.0)
        case (false, true, true) => Array(0.0, -1.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0)
        case (false, true, false) => Array(-1.0, 0.0, -1.0, 0.0, 0.0, -1.0, -1.0, 0.0)
        case (false, false, true) => Array(0.0, -1.0, 0.0, -1.0, 0.0, -1.0, -1.0, 0.0)
        case (false, false, false) => Array(-1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, -1.0)
      }
    }
  }
}
