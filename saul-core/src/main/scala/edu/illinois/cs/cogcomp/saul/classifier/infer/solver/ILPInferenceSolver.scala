/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.solver

import edu.illinois.cs.cogcomp.infer.ilp.{ GurobiHook, ILPSolver, OJalgoHook }
import edu.illinois.cs.cogcomp.lbjava.classify.ScoreSet
import edu.illinois.cs.cogcomp.lbjava.infer.BalasHook
import edu.illinois.cs.cogcomp.saul.classifier.infer._
import edu.illinois.cs.cogcomp.saul.lbjrelated.LBJLearnerEquivalent
import edu.illinois.cs.cogcomp.saul.util.Logging

import scala.collection.{ Seq, mutable }

class ILPInferenceSolver[T <: AnyRef, HEAD <: AnyRef](
  val ilpSolverType: SolverType,
  val optimizationType: OptimizationType,
  val onClassifier: LBJLearnerEquivalent,
  val useCaching: Boolean = false
)
  extends InferenceSolver[T, HEAD] with Logging {

  private val inferenceManager = new ILPInferenceManager()

  def solve(constraintsOpt: Option[Constraint[_]], priorAssignment: Seq[Assignment]): Seq[Assignment] = {
    val instances = getInstancesInvolvedInProblem(constraintsOpt)

    /** The following cache-key is very important, as it defines what to and when to cache the results of the inference.
      * The first term encodes the instances involved in the constraint, after propositionalization, and the second term
      * contains pure definition of the constraint before any propositionalization.
      */
    val cacheKey = instances.map(_.toString).toSeq.sorted.mkString("*") + constraintsOpt

    val resultOpt = if (useCaching) ILPInferenceManager.cachedResults.get(cacheKey) else None
    resultOpt match {
      case Some((cachedSolver, cachedClassifier, cachedEstimatorToSolverLabelMap)) =>
        getInstanceAssignment(priorAssignment, cachedSolver, cachedEstimatorToSolverLabelMap)
      case None =>
        logger.trace("Solving a new inference problem")

        // create a new solver instance
        val solver = getSolverInstance
        solver.setMaximize(optimizationType == Max)

        // populate the instances connected to head
        priorAssignment.foreach(inferenceManager.addVariablesToInferenceProblem(_, solver))

        constraintsOpt.foreach { constraints =>
          val inequalities = inferenceManager.processConstraints(constraints, solver)
          inequalities.foreach { ineq =>
            solver.addGreaterThanConstraint(ineq.x, ineq.a, ineq.b)
          }
        }

        solver.solve()

        if (!solver.isSolved) {
          logger.warn("Instance not solved . . . ")
        }

        if (useCaching) {
          ILPInferenceManager.cachedResults.put(cacheKey, (solver, onClassifier, inferenceManager.estimatorToSolverLabelMap))
        }

        getInstanceAssignment(priorAssignment, solver, inferenceManager.estimatorToSolverLabelMap)
    }
  }

  /** given a solver type, instantiates a solver, upon calling it */
  private def getSolverInstance: ILPSolver = ilpSolverType match {
    case OJAlgo => new OJalgoHook()
    case Gurobi => new GurobiHook()
    case Balas => new BalasHook()
    case _ => throw new Exception("Hook not found! ")
  }

  /** given an instance, the result of the inference inside an [[ILPSolver]], and a hashmap which connects
    * classifier labels to solver's internal variables, returns a label for a given instance
    */
  private def getInstanceLabel(t: T, solver: ILPSolver,
    classifier: LBJLearnerEquivalent,
    estimatorToSolverLabelMap: mutable.Map[LBJLearnerEquivalent, mutable.Map[_, Seq[(Int, String)]]]): String = {
    val estimatorSpecificMap = estimatorToSolverLabelMap(classifier).asInstanceOf[mutable.Map[T, Seq[(Int, String)]]]
    estimatorSpecificMap.get(t) match {
      case Some(indexLabelPairs) =>
        val values = indexLabelPairs.map {
          case (ind, _) => solver.getIntegerValue(ind)
        }
        // exactly one label should be active; if not, [probably] the inference has been infeasible and
        // it is not usable, in which case we make direct calls to the non-constrained classifier.
        if (values.sum == 1) {
          indexLabelPairs.collectFirst {
            case (ind, label) if solver.getIntegerValue(ind) == 1.0 => label
          }.get
        } else {
          classifier.classifier.scores(t).highScoreValue()
        }
      case None => throw new Exception("instance is not cached ... weird! :-/ ")
    }
  }

  private def getInstanceAssignment(
    priorAssignments: Seq[Assignment],
    solver: ILPSolver,
    estimatorToSolverLabelMap: mutable.Map[LBJLearnerEquivalent, mutable.Map[_, Seq[(Int, String)]]]
  ): Seq[Assignment] = {
    priorAssignments.foreach({ assignment: Assignment =>
      val estimatorSpecificMap = estimatorToSolverLabelMap(assignment.learner).asInstanceOf[mutable.Map[Any, Seq[(Int, String)]]]
      assignment.foreach({
        case (instance: Any, scores: ScoreSet) =>
          val scoresArray = scores.toArray
          estimatorSpecificMap.get(instance) match {
            case Some(indexLabelPairs) =>
              val values = indexLabelPairs.map {
                case (ind, _) => solver.getIntegerValue(ind)
              }
              // exactly one label should be active; if not, [probably] the inference has been infeasible and
              // it is not usable, in which case we make direct calls to the non-constrained classifier.
              if (values.sum == 1) {
                val instanceLabel = indexLabelPairs.collectFirst {
                  case (ind, label) if solver.getIntegerValue(ind) == 1.0 => label
                }.get
                scoresArray.foreach({ score =>
                  score.score = if (score.value == instanceLabel) 1.0 else 0.0
                })
              }

            case None => throw new Exception("instance is not cached ... weird! :-/ ")
          }
      })
    })
    priorAssignments
  }

  private def getInstancesInvolvedInProblem(constraintsOpt: Option[Constraint[_]]): Option[Set[_]] = {
    constraintsOpt.map { constraint => getInstancesInvolved(constraint) }
  }

  /** find all the instances used in the definition of the constraint.
    * This is used in caching the results of inference
    */
  private def getInstancesInvolved(constraint: Constraint[_]): Set[_] = {
    constraint match {
      case c: PropositionalEqualityConstraint[_] =>
        Set(c.instanceOpt.get)
      case c: PairConjunction[_, _] =>
        getInstancesInvolved(c.c1) ++ getInstancesInvolved(c.c2)
      case c: PairDisjunction[_, _] =>
        getInstancesInvolved(c.c1) ++ getInstancesInvolved(c.c2)
      case c: Negation[_] =>
        getInstancesInvolved(c.p)
      case c: ConstraintCollections[_, _] =>
        c.constraints.foldRight(Set[Any]()) {
          case (singleConstraint, ins) =>
            ins union getInstancesInvolved(singleConstraint).asInstanceOf[Set[Any]]
        }
      case c: EstimatorPairEqualityConstraint[_] =>
        Set(c.instance)
      case c: InstancePairEqualityConstraint[_] =>
        Set(c.instance1, c.instance2Opt.get)
      case _ =>
        throw new Exception("Unknown constraint exception! This constraint should have been rewritten in terms of other constraints. ")
    }
  }
}
