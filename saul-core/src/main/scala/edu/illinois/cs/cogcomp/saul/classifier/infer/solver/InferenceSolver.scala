/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.solver

import edu.illinois.cs.cogcomp.saul.classifier.infer.{ Assignment, Constraint }

import scala.collection.Seq

sealed trait OptimizationType
case object Max extends OptimizationType
case object Min extends OptimizationType

trait InferenceSolver[T <: AnyRef, HEAD <: AnyRef] {
  def solve(constraintsOpt: Option[Constraint[_]], priorAssignment: Seq[Assignment]): Seq[Assignment]
}
