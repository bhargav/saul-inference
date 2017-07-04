/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.solver

import edu.illinois.cs.cogcomp.infer.ilp.{ BeamSearch, GurobiHook, OJalgoHook }
import edu.illinois.cs.cogcomp.lbjava.classify.{ Score, ScoreSet }
import edu.illinois.cs.cogcomp.lbjava.infer.ad3.AD3Inference
import edu.illinois.cs.cogcomp.lbjava.infer.maxbp.MaxBPInference
import edu.illinois.cs.cogcomp.lbjava.infer.srmpsolver.SRMPInference
import edu.illinois.cs.cogcomp.lbjava.infer._
import edu.illinois.cs.cogcomp.saul.classifier.infer.{ Assignment, Constraint }
import edu.illinois.cs.cogcomp.saul.util.Logging

import scala.collection.{ Seq, mutable }

/** possible solvers to use */
sealed trait SolverType
case object Gurobi extends SolverType
case object OJAlgo extends SolverType
case object Balas extends SolverType
case object BeamSearchSolver extends SolverType
case object AD3 extends SolverType
case object MaxBP extends SolverType
case object SRMP extends SolverType

final class AD3InferenceSolver extends AD3Inference with PropositionalInference {
  override def addConstraint(c: PropositionalConstraint, variablesToConsider: Seq[FirstOrderVariable]): Unit = {
    if (constraint == null)
      constraint = c
    else
      constraint = new PropositionalConjunction(constraint.asInstanceOf[PropositionalConstraint], c)

    // Calling getVariable adds the variable to the `variables` field that is used in the `infer` method.
    variablesToConsider.foreach(getVariable)
  }
}

final class MaxBPInferenceSolver extends MaxBPInference with PropositionalInference {
  override def addConstraint(c: PropositionalConstraint, variablesToConsider: Seq[FirstOrderVariable]): Unit = {
    if (constraint == null)
      constraint = c
    else
      constraint = new PropositionalConjunction(constraint.asInstanceOf[PropositionalConstraint], c)

    // Calling getVariable adds the variable to the `variables` field that is used in the `infer` method.
    variablesToConsider.foreach(getVariable)
  }
}

final class SRMPInferenceSolver extends SRMPInference with PropositionalInference {
  override def addConstraint(c: PropositionalConstraint, variablesToConsider: Seq[FirstOrderVariable]): Unit = {
    if (constraint == null)
      constraint = c
    else
      constraint = new PropositionalConjunction(constraint.asInstanceOf[PropositionalConstraint], c)

    // Calling getVariable adds the variable to the `variables` field that is used in the `infer` method.
    variablesToConsider.foreach(getVariable)
  }
}

object InferenceSolverFactory {
  def getSolver(solverType: SolverType): InferenceSolver = {
    new InferenceSolver with Logging {
      override def solve(constraintsOpt: Option[Constraint[_]], priorAssignment: Seq[Assignment]): Seq[Assignment] = {
        val variableBuffer = mutable.HashSet[FirstOrderVariable]()
        val lbjConstraints = LBJavaConstraintUtilities.transformToLBJConstraint(constraintsOpt.get, variableBuffer)

        val solver = getSolverInstance(solverType)
        solver.addConstraint(lbjConstraints, variableBuffer.toSeq)

        val finalAssignment = priorAssignment.map({ assignment: Assignment =>
          val finalAssgn = Assignment(assignment.learner)

          assignment.foreach({
            case (instance: Any, scoreset: ScoreSet) =>
              logger.debug(s"$instance - Previous: ${scoreset.highScoreValue()}")

              val newScoreSet = solver match {
                case solverWithScores: Inference with InferenceWithScores with PropositionalInference =>
                  solverWithScores.scoresOf(assignment.learner.classifier, instance)
                case _ =>
                  val label = solver.valueOf(assignment.learner.classifier, instance)
                  val newScores = scoreset.toArray
                    .map({ score: Score =>
                      new Score(score.value, if (score.value == label) 1.0 else 0.0)
                    })
                  new ScoreSet(newScores)
              }

              assert(newScoreSet.toArray.map(_.score).sum == 1.0)
              finalAssgn += (instance -> newScoreSet)

              logger.debug(s"$instance - Previous: ${newScoreSet.highScoreValue()}")
          })

          finalAssgn
        })

        finalAssignment
      }
    }
  }

  private def getSolverInstance(solverType: SolverType): Inference with PropositionalInference = {
    solverType match {
      case OJAlgo => new LBJavaPropositionalILPInference(new OJalgoHook())
      case Gurobi => new LBJavaPropositionalILPInference(new GurobiHook())
      case Balas => new LBJavaPropositionalILPInference(new BalasHook())
      case BeamSearchSolver => new LBJavaPropositionalILPInference(new BeamSearch(5))
      case AD3 => new AD3InferenceSolver()
      case MaxBP => new MaxBPInferenceSolver()
      case SRMP => new SRMPInferenceSolver()
      case _ => throw new Exception("Solver not found!")
    }
  }
}
