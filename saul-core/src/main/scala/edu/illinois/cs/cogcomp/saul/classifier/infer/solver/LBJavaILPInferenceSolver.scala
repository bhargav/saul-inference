/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.solver

import java.util

import edu.illinois.cs.cogcomp.infer.ilp.{ GurobiHook, OJalgoHook }
import edu.illinois.cs.cogcomp.lbjava.classify.{ Score, ScoreSet }
import edu.illinois.cs.cogcomp.lbjava.infer.{ Constraint => _, _ }
import edu.illinois.cs.cogcomp.saul.classifier.infer._
import edu.illinois.cs.cogcomp.saul.util.Logging

class LBJavaILPInferenceSolver[T <: AnyRef, HEAD <: AnyRef] extends InferenceSolver[T, HEAD] with Logging {
  private def transformToLBJConstraint(constraint: Constraint[_]): FirstOrderConstraint = {
    constraint match {
      case c: PropositionalEqualityConstraint[_] =>
        val variable = new FirstOrderVariable(c.estimator.classifier, c.instanceOpt.get)
        val value = c.equalityValOpt.orElse(c.inequalityValOpt).get
        val isEquality = c.equalityValOpt.nonEmpty
        new FirstOrderEqualityWithValue(isEquality, variable, value)
      case c: PairConjunction[_, _] =>
        val leftConstraint = transformToLBJConstraint(c.c1)
        val rightConstraint = transformToLBJConstraint(c.c2)
        new FirstOrderConjunction(leftConstraint, rightConstraint)
      case c: PairDisjunction[_, _] =>
        val leftConstraint = transformToLBJConstraint(c.c1)
        val rightConstraint = transformToLBJConstraint(c.c2)
        new FirstOrderDisjunction(leftConstraint, rightConstraint)
      case c: Implication[_, _] =>
        val leftConstraint = transformToLBJConstraint(c.p)
        val rightConstraint = transformToLBJConstraint(c.q)
        new FirstOrderImplication(leftConstraint, rightConstraint)
      case c: Negation[_] =>
        val constraint = transformToLBJConstraint(c.p)
        logger.warn("Unsupported!")
        // Verify this once.
        new FirstOrderNegation(constraint)
      case c: ForAll[_, _] =>
        val firstOrderConstraints = c.constraints.map({ cons: Constraint[_] =>
          transformToLBJConstraint(cons)
        }).toList

        firstOrderConstraints.length match {
          case 0 => new FirstOrderConstant(true)
          case 1 => firstOrderConstraints.head
          case _ => {
            val first = firstOrderConstraints.head
            val second = firstOrderConstraints(1)
            val forAll = new FirstOrderConjunction(first, second)
            firstOrderConstraints.drop(2).foreach(forAll.add)
            forAll
          }
        }

      case c: AtLeast[_, _] =>
        logger.warn("Unsupported!")
        new FirstOrderConstant(true)
      case c: AtMost[_, _] =>
        logger.warn("Unsupported!")
        new FirstOrderConstant(true)
      case c: Exactly[_, _] =>
        logger.warn("Unsupported!")
        new FirstOrderConstant(true)
      case c: ConstraintCollections[_, _] =>
        logger.warn("Unsupported!")
        new FirstOrderConstant(true)
      //        c.constraints.foldRight(Set[Any]()) {
      //          case (singleConstraint, ins) =>
      //            ins union getInstancesInvolved(singleConstraint).asInstanceOf[Set[Any]]
      //        }
      case _ =>
        //        new FirstOrderConstant(true)
        //      case c: EstimatorPairEqualityConstraint[_] =>
        //        Set(c.instance)
        //      case c: InstancePairEqualityConstraint[_] =>
        //        Set(c.instance1, c.instance2Opt.get)
        //      case _ =>
        throw new Exception("Unknown constraint exception! This constraint should have been rewritten in terms of other constraints. ")
    }
  }

  override def solve(constraintsOpt: Option[Constraint[_]], priorAssignment: Seq[Assignment]): Seq[Assignment] = {
    val lbjConstraints = transformToLBJConstraint(constraintsOpt.get)

    val inference = new LBJavaPropositionalILPInference(new OJalgoHook())
    inference.addConstraint(lbjConstraints.propositionalize())

    val finalAssignment = priorAssignment.map({ assignment: Assignment =>
      val finalAssgn = Assignment(assignment.learner)

      assignment.foreach({
        case (instance: Any, scoreset: ScoreSet) =>
          //        println(s"$instance - Previous: ${scoreset.highScoreValue()} - After ${inference.valueOf(assignment.learner.classifier, instance)}")
          val label = inference.valueOf(assignment.learner.classifier, instance)
          val allLabels = scoreset.toArray
          allLabels.foreach({ score: Score =>
            if (score.value == label) {
              score.score = 1.0
            } else {
              score.score = 0.0
            }
          })

          finalAssgn += (instance -> new ScoreSet(allLabels))
      })

      finalAssgn
    })

    finalAssignment
  }
}
