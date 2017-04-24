/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.solver

import edu.illinois.cs.cogcomp.infer.ilp.{ GurobiHook, ILPSolver, OJalgoHook }
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

  def solve(cacheKey: String, instance: T, constraintsOpt: Option[Constraint[_]], candidates: Seq[T]): String = {
    val resultOpt = if (useCaching) ILPInferenceManager.cachedResults.get(cacheKey) else None
    resultOpt match {
      case Some((cachedSolver, cachedClassifier, cachedEstimatorToSolverLabelMap)) =>
        getInstanceLabel(instance, cachedSolver, onClassifier, cachedEstimatorToSolverLabelMap)
      case None =>
        // create a new solver instance
        val solver = getSolverInstance
        solver.setMaximize(optimizationType == Max)

        // populate the instances connected to head
        inferenceManager.addVariablesToInferenceProblem(candidates, onClassifier, solver)

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

        getInstanceLabel(instance, solver, onClassifier, inferenceManager.estimatorToSolverLabelMap)
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
}
