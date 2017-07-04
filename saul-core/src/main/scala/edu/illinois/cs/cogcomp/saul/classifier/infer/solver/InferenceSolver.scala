/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.solver

import edu.illinois.cs.cogcomp.lbjava.infer.{ FirstOrderVariable, Inference, PropositionalConstraint }
import edu.illinois.cs.cogcomp.saul.classifier.infer.{ Assignment, Constraint }

import scala.collection.Seq

sealed trait OptimizationType
case object Max extends OptimizationType
case object Min extends OptimizationType

trait InferenceSolver {
  def solve(constraintsOpt: Option[Constraint[_]], priorAssignment: Seq[Assignment]): Seq[Assignment]
}

trait PropositionalInference extends Inference {
  /** Add a Propositional Constraint to the system of constraints to solve.
    * @param c Propositional constraint to consider.
    */
  def addConstraint(c: PropositionalConstraint, variablesToConsider: Seq[FirstOrderVariable]): Unit
}