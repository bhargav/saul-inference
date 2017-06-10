/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.solver

import edu.illinois.cs.cogcomp.lbjava.classify.{ Score, ScoreSet }
import edu.illinois.cs.cogcomp.lbjava.infer.{ PropositionalConstraint => LBJPropositionalConstraint, _ }

import edu.illinois.cs.cogcomp.saul.classifier.infer.{ Constraint => SaulConstraint, _ }
import edu.illinois.cs.cogcomp.saul.util.Logging

import scala.collection.mutable

class LBJavaILPInferenceSolver[T <: AnyRef, HEAD <: AnyRef](solverType: SolverType)
  extends InferenceSolver[T, HEAD] with Logging {

  override def solve(constraintsOpt: Option[SaulConstraint[_]], priorAssignment: Seq[Assignment]): Seq[Assignment] = {
    val solverHookInstance = ILPInferenceSolver.getSolverInstance(solverType)
    val inference = new LBJavaPropositionalILPInference(solverHookInstance)
    val variableBuffer = mutable.HashSet[FirstOrderVariable]()
    val lbjConstraints = transformToLBJConstraint(constraintsOpt.get, variableBuffer)

    inference.addConstraint(lbjConstraints, variableBuffer.toSeq)
    inference.infer()

    val finalAssignment = priorAssignment.map({ assignment: Assignment =>
      val finalAssgn = Assignment(assignment.learner)

      assignment.foreach({
        case (instance: Any, scoreset: ScoreSet) =>
          logger.debug(s"$instance - Previous: ${scoreset.highScoreValue()}")
          val label = inference.valueOf(assignment.learner.classifier, instance)
          logger.debug(s"$instance - After ${inference.valueOf(assignment.learner.classifier, instance)}")

          val newScores = scoreset.toArray
            .map({ score: Score =>
              new Score(score.value, if (score.value == label) 1.0 else 0.0)
            })

          assert(newScores.map(_.score).sum == 1.0)
          finalAssgn += (instance -> new ScoreSet(newScores))
      })

      finalAssgn
    })

    finalAssignment
  }

  private def transformToLBJConstraint(constraint: SaulConstraint[_], variableSet: mutable.HashSet[FirstOrderVariable]): LBJPropositionalConstraint = {
    constraint match {
      case c: PropositionalEqualityConstraint[_] =>
        val variable = new FirstOrderVariable(c.estimator.classifier, c.instanceOpt.get)
        variableSet += variable

        val value = c.equalityValOpt.orElse(c.inequalityValOpt).get
        val isEquality = c.equalityValOpt.nonEmpty
        new FirstOrderEqualityWithValue(isEquality, variable, value).propositionalize()

      case c: InstancePairEqualityConstraint[_] =>
        val firstVariable = new FirstOrderVariable(c.estimator.classifier, c.instance1)
        val secondVariable = new FirstOrderVariable(c.estimator.classifier, c.instance2Opt.get)
        variableSet += firstVariable
        variableSet += secondVariable

        new FirstOrderEqualityWithVariable(c.equalsOpt.get, firstVariable, secondVariable).propositionalize()

      case c: EstimatorPairEqualityConstraint[_] =>
        val firstVariable = new FirstOrderVariable(c.estimator1.classifier, c.instance)
        val secondVariable = new FirstOrderVariable(c.estimator2Opt.get.classifier, c.instance)
        variableSet += firstVariable
        variableSet += secondVariable

        new FirstOrderEqualityWithVariable(c.equalsOpt.get, firstVariable, secondVariable).propositionalize()

      case c: PairConjunction[_, _] =>
        val leftConstraint = transformToLBJConstraint(c.c1, variableSet)
        val rightConstraint = transformToLBJConstraint(c.c2, variableSet)
        new PropositionalConjunction(leftConstraint, rightConstraint)

      case c: PairDisjunction[_, _] =>
        val leftConstraint = transformToLBJConstraint(c.c1, variableSet)
        val rightConstraint = transformToLBJConstraint(c.c2, variableSet)
        new PropositionalDisjunction(leftConstraint, rightConstraint)

      case c: Implication[_, _] =>
        val leftConstraint = transformToLBJConstraint(c.p, variableSet)
        val rightConstraint = transformToLBJConstraint(c.q, variableSet)
        new PropositionalImplication(leftConstraint, rightConstraint)

      case c: Negation[_] =>
        val constraint = transformToLBJConstraint(c.p, variableSet)
        logger.warn("Unsupported!")
        // Verify this once.
        new PropositionalNegation(constraint)

      case c: ForAll[_, _] =>
        val firstOrderConstraints = c.constraints
          .map({ cons: SaulConstraint[_] =>
            transformToLBJConstraint(cons, variableSet)
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
        val propConstraints = c.constraints.map(c => transformToLBJConstraint(c, variableSet)).toArray
        new PropositionalAtLeast(propConstraints, c.k)

      case c: AtMost[_, _] =>
        // AtMost can be written as AtLeast of (length - k) size.
        val negativePropConstraints: Array[LBJPropositionalConstraint] = c.constraints
          .map(c => new PropositionalNegation(transformToLBJConstraint(c, variableSet)))
          .toArray
        new PropositionalAtLeast(negativePropConstraints, negativePropConstraints.length - c.k)

      case c: Exactly[_, _] =>
        // Exactly can be written as a combination of AtLeast and AtMost
        val propositionalConstraints = c.constraints.map(c => transformToLBJConstraint(c, variableSet)).toArray
        val negativePropConstraints: Array[LBJPropositionalConstraint] = c.constraints
          .map(c => new PropositionalNegation(transformToLBJConstraint(c, variableSet)))
          .toArray
        val atLeastConstraint = new PropositionalAtLeast(propositionalConstraints, c.k)
        val atMostConstraint = new PropositionalAtLeast(negativePropConstraints, negativePropConstraints.length - c.k)
        new PropositionalConjunction(atLeastConstraint, atMostConstraint)

      case _ =>
        throw new Exception("Unknown constraint exception! This constraint should have been rewritten in terms of other constraints. ")
    }
  }

}
