/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.solver

import edu.illinois.cs.cogcomp.infer.ilp.{ ILPSolver, OJalgoHook }
import edu.illinois.cs.cogcomp.lbjava.classify.{ Score, ScoreSet }
import edu.illinois.cs.cogcomp.lbjava.infer.{ PropositionalConstraint => LBJPropositionalConstraint, _ }
import edu.illinois.cs.cogcomp.saul.classifier.infer.{ Constraint => SaulConstraint, _ }
import edu.illinois.cs.cogcomp.saul.util.Logging

class LBJavaILPInferenceSolver[T <: AnyRef, HEAD <: AnyRef](solverHookInstance: ILPSolver = new OJalgoHook)
  extends InferenceSolver[T, HEAD] with Logging {
  private def transformToLBJConstraint(constraint: SaulConstraint[_]): LBJPropositionalConstraint = {
    constraint match {
      case c: PropositionalEqualityConstraint[_] =>
        val variable = new FirstOrderVariable(c.estimator.classifier, c.instanceOpt.get)
        val value = c.equalityValOpt.orElse(c.inequalityValOpt).get
        val isEquality = c.equalityValOpt.nonEmpty
        new FirstOrderEqualityWithValue(isEquality, variable, value).propositionalize()

      case c: PairConjunction[_, _] =>
        val leftConstraint = transformToLBJConstraint(c.c1)
        val rightConstraint = transformToLBJConstraint(c.c2)
        new PropositionalConjunction(leftConstraint, rightConstraint)

      case c: PairDisjunction[_, _] =>
        val leftConstraint = transformToLBJConstraint(c.c1)
        val rightConstraint = transformToLBJConstraint(c.c2)
        new PropositionalDisjunction(leftConstraint, rightConstraint)

      case c: Implication[_, _] =>
        val leftConstraint = transformToLBJConstraint(c.p)
        val rightConstraint = transformToLBJConstraint(c.q)
        new PropositionalImplication(leftConstraint, rightConstraint)

      case c: Negation[_] =>
        val constraint = transformToLBJConstraint(c.p)
        logger.warn("Unsupported!")
        // Verify this once.
        new PropositionalNegation(constraint)

      case c: ForAll[_, _] =>
        val firstOrderConstraints = c.constraints
          .map({ cons: SaulConstraint[_] =>
            transformToLBJConstraint(cons)
          }).toList

        firstOrderConstraints.length match {
          case 0 => new PropositionalConstant(true)
          case 1 => firstOrderConstraints.head
          case _ =>
            val first = firstOrderConstraints.head
            val second = firstOrderConstraints(1)
            val forAll = new PropositionalConjunction(first, second)
            firstOrderConstraints.drop(2).foreach(forAll.add)
            forAll
        }

      case c: AtLeast[_, _] =>
        val propConstraints = c.constraints.map(transformToLBJConstraint).toArray
        new PropositionalAtLeast(propConstraints, c.k)

      case c: AtMost[_, _] =>
        // AtMost can be written as AtLeast of (length - k) size.
        val negativePropConstraints: Array[LBJPropositionalConstraint] = c.constraints
          .map(c => new PropositionalNegation(transformToLBJConstraint(c)))
          .toArray
        new PropositionalAtLeast(negativePropConstraints, negativePropConstraints.length - c.k)

      case c: Exactly[_, _] =>
        // Exactly can be written as a combination of AtLeast and AtMost
        val propositionalConstraints = c.constraints.map(transformToLBJConstraint).toArray
        val negativePropConstraints: Array[LBJPropositionalConstraint] = c.constraints
          .map(c => new PropositionalNegation(transformToLBJConstraint(c)))
          .toArray
        val atLeastConstraint = new PropositionalAtLeast(propositionalConstraints, c.k)
        val atMostConstraint = new PropositionalAtLeast(negativePropConstraints, negativePropConstraints.length - c.k)
        new PropositionalConjunction(atLeastConstraint, atMostConstraint)

      case _ =>
        throw new Exception("Unknown constraint exception! This constraint should have been rewritten in terms of other constraints. ")
    }
  }

  override def solve(constraintsOpt: Option[SaulConstraint[_]], priorAssignment: Seq[Assignment]): Seq[Assignment] = {
    val inference = new LBJavaPropositionalILPInference(solverHookInstance)
    val lbjConstraints = transformToLBJConstraint(constraintsOpt.get)

    inference.addConstraint(lbjConstraints)

    val finalAssignment = priorAssignment.map({ assignment: Assignment =>
      val finalAssgn = Assignment(assignment.learner)

      assignment.foreach({
        case (instance: Any, scoreset: ScoreSet) =>
          logger.debug(s"$instance - Previous: ${scoreset.highScoreValue()}")
          val label = inference.valueOf(assignment.learner.classifier, instance)
          logger.debug(s"$instance - After ${inference.valueOf(assignment.learner.classifier, instance)}")

          val allLabels = scoreset.toArray
          allLabels.foreach({ score: Score =>
            if (score.value == label) {
              score.score = 1.0
            } else {
              score.score = 0.0
            }
          })

          assert(allLabels.map(_.score).sum == 1.0)
          finalAssgn += (instance -> new ScoreSet(allLabels))
      })

      finalAssgn
    })

    finalAssignment
  }
}
