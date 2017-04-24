/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saulexamples.Badge

import edu.illinois.cs.cogcomp.infer.ilp.OJalgoHook
import edu.illinois.cs.cogcomp.saul.classifier.infer.ConstrainedClassifier
import edu.illinois.cs.cogcomp.saul.classifier.infer.Constraint._
import edu.illinois.cs.cogcomp.saul.classifier.infer.solver.Gurobi
import edu.illinois.cs.cogcomp.saulexamples.Badge.BadgeClassifiers.{ BadgeClassifier, BadgeClassifierMulti, BadgeOppositClassifier, BadgeOppositClassifierMulti }

/** Created by Parisa on 11/1/16.
  */
object BadgeConstrainedClassifiers {

  def binaryConstraint = BadgeDataModel.badge.ForEach { x: String =>
    (BadgeClassifier on x is "negative") ==> (BadgeOppositClassifier on x is "positive")
  }

  def binaryConstraintOverMultiClassifiers = BadgeDataModel.badge.ForEach { x: String =>
    (BadgeClassifierMulti on x is "negative") ==> (BadgeOppositClassifierMulti on x is "positive")
  }

  object badgeConstrainedClassifier extends ConstrainedClassifier[String, String] {
    override def subjectTo = Some(binaryConstraint)
    override def solverType = Gurobi
    override lazy val onClassifier = BadgeClassifier
  }

  object oppositBadgeConstrainedClassifier extends ConstrainedClassifier[String, String] {
    override def subjectTo = Some(binaryConstraint)
    override def solverType = Gurobi
    override lazy val onClassifier = BadgeOppositClassifier
  }

  object badgeConstrainedClassifierMulti extends ConstrainedClassifier[String, String] {
    override def subjectTo = Some(binaryConstraintOverMultiClassifiers)
    override def solverType = Gurobi
    override lazy val onClassifier = BadgeClassifierMulti
  }

  object oppositBadgeConstrainedClassifierMulti extends ConstrainedClassifier[String, String] {
    override def subjectTo = Some(binaryConstraintOverMultiClassifiers)
    override def solverType = Gurobi
    override lazy val onClassifier = BadgeOppositClassifierMulti
  }
}
