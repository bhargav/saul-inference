/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.solver

import edu.illinois.cs.cogcomp.lbjava.classify.{ Score, ScoreSet }
import edu.illinois.cs.cogcomp.lbjava.learn.Softmax
import edu.illinois.cs.cogcomp.saul.classifier.infer._
import edu.illinois.cs.cogcomp.saul.classifier.infer.factorgraph.SRMPFactorUtils
import edu.illinois.cs.cogcomp.saul.lbjrelated.LBJLearnerEquivalent
import edu.illinois.cs.cogcomp.saul.util.Logging
import srmp.{ EnergyOptions, Factor, FactorType, SolverType => SRMPSolverType, Energy => FactorGraph, Node => FactorNode }

import scala.collection.mutable

final class SRMPInferenceSolver[T <: AnyRef, HEAD <: AnyRef] extends InferenceSolver[T, HEAD] with Logging {
  override def solve(constraintsOpt: Option[Constraint[_]], priorAssignment: Seq[Assignment]): Seq[Assignment] = {
    val softmax = new Softmax()
    val classifierLabelMap = new mutable.HashMap[LBJLearnerEquivalent, List[(String, Int)]]()
    val instanceVariableMap = new mutable.HashMap[(LBJLearnerEquivalent, String, Any), (FactorNode, Boolean)]()

    // XXX - Get appropriate number of nodes
    val factorGraph = new FactorGraph(50)

    val factors = new mutable.ListBuffer[Factor]()
    val variables = new mutable.ListBuffer[FactorNode]()

    priorAssignment.foreach({ assignment: Assignment =>
      val labels: List[String] = assignment.learner.classifier.scores(assignment.head._1).toArray.map(_.value).toList.sorted

      val labelIndexMap = if (labels.size == 2) {
        labels.zip(Array(0, 1))
      } else {
        labels.zip(Array.fill(labels.size)(1))
      }

      classifierLabelMap += (assignment.learner -> labelIndexMap)

      assignment.foreach({
        case (instance: Any, scores: ScoreSet) =>
          val normalizedScoreset = softmax.normalize(scores)

          if (labels.size == 2) {
            val costs = labelIndexMap.map({
              case (label: String, _) =>
                /* Normalized Probability Scores */
                -1 * normalizedScoreset.getScore(label).score
            }).toArray

            val node = factorGraph.addNode(2, costs)
            labelIndexMap.foreach({
              case (label: String, idx: Int) =>
                instanceVariableMap += ((assignment.learner, label, instance) -> (node, idx == 1))
            })
          } else {
            logger.info("Multi-variables thingy")
            val multiNodes = labelIndexMap.map({
              case (label: String, _) =>
                /* Normalized Probability Scores */
                val score = -1 * normalizedScoreset.getScore(label).score
                val node = factorGraph.addNode(2, Array(-1 - score, score))

                instanceVariableMap += ((assignment.learner, label, instance) -> (node, true))
                node
            })

            // XXX - Add an XOR factor for multiNodes
          }
      })
    })

    if (constraintsOpt.nonEmpty) {
      //      logger.info("Processing constraints")
      processConstraints(constraintsOpt.get, instanceVariableMap, factorGraph, factors, variables)
    }

    val solverOptions = new EnergyOptions()
    solverOptions.setSolverType(SRMPSolverType.SRMP)
    solverOptions.setVerbose(false)

    val lowerBound = factorGraph.solve(solverOptions)
    //    logger.info(s"Lower bound = $lowerBound")

    if (lowerBound == Double.NegativeInfinity) {
      logger.warn("Unsolved problem. Using original assignment.")
      priorAssignment
    } else {
      val finalAssignments = priorAssignment.map({ assignment: Assignment =>
        val finalAssgn = Assignment(assignment.learner)
        val domain = classifierLabelMap(assignment.learner)

        assignment.foreach({
          case (instance: Any, _) =>
            val newScores = domain.map({
              case (label: String, idx: Int) =>
                val variableTuple = instanceVariableMap((assignment.learner, label, instance))
                val binaryNode = variableTuple._1
                val posterior = factorGraph.getSolution(binaryNode)

                new Score(label, if (posterior == idx) 1.0 else 0.0)
            }).toArray

            assert(newScores.map(_.score).sum == 1, "A labels' scores should sum to 1 after inference.")
            finalAssgn += (instance -> new ScoreSet(newScores))
        })

        finalAssgn
      })

      finalAssignments
    }
  }

  private def getOutputVariable(factorGraph: FactorGraph, state: Boolean): FactorNode = {
    factorGraph.addNode(2, Array.fill(2)(0))
  }

  private def processConstraints(
    constraint: Constraint[_],
    instanceVariableMap: mutable.HashMap[(LBJLearnerEquivalent, String, Any), (FactorNode, Boolean)],
    factorGraph: FactorGraph,
    factors: mutable.ListBuffer[Factor],
    variables: mutable.ListBuffer[FactorNode],
    createVariable: Boolean = false
  ): Option[(FactorNode, Boolean)] = {
    constraint match {
      case c: PropositionalEqualityConstraint[_] =>
        val value = c.equalityValOpt.orElse(c.inequalityValOpt).get
        val variableWithState = instanceVariableMap((c.estimator, value, c.instanceOpt.get))

        // Handle negative variable states
        val isEquality = c.equalityValOpt.nonEmpty == variableWithState._2

        Some((variableWithState._1, isEquality))
      case c: PairConjunction[_, _] =>
        val leftVariable = processConstraints(
          c.c1,
          instanceVariableMap, factorGraph, factors, variables, createVariable = true
        ).get
        val rightVariable = processConstraints(
          c.c2,
          instanceVariableMap, factorGraph, factors, variables, createVariable = true
        ).get

        if (createVariable) {
          val outputVariable = getOutputVariable(factorGraph, state = true)
          val costs = SRMPFactorUtils.getPairConjunctionCosts(leftVariable._2, rightVariable._2, Some(true))
          val factor = factorGraph.addFactor(3, Array(leftVariable._1, rightVariable._1, outputVariable), costs, FactorType.GeneralFactorType)

          //          logger.info("Adding a factor")

          factors += factor
          variables += outputVariable

          Some((outputVariable, true))
        } else {
          val costs = SRMPFactorUtils.getPairConjunctionCosts(leftVariable._2, rightVariable._2)
          val factor = factorGraph.addPairwiseFactor(leftVariable._1, rightVariable._1, costs)

          //          logger.info("Adding a pairwise factor")

          factors += factor
          None
        }
      case c: PairDisjunction[_, _] =>
        val leftVariable = processConstraints(
          c.c1,
          instanceVariableMap, factorGraph, factors, variables, createVariable = true
        ).get
        val rightVariable = processConstraints(
          c.c2,
          instanceVariableMap, factorGraph, factors, variables, createVariable = true
        ).get

        if (createVariable) {
          val outputVariable = getOutputVariable(factorGraph, state = true)
          val costs = SRMPFactorUtils.getPairDisjunctionCosts(leftVariable._2, rightVariable._2, Some(true))
          val factor = factorGraph.addFactor(3, Array(leftVariable._1, rightVariable._1, outputVariable), costs, FactorType.GeneralFactorType)

          //          logger.info("Adding a factor")

          factors += factor
          variables += outputVariable

          Some((outputVariable, true))
        } else {
          val costs = SRMPFactorUtils.getPairDisjunctionCosts(leftVariable._2, rightVariable._2)
          val factor = factorGraph.addPairwiseFactor(leftVariable._1, rightVariable._1, costs)

          //          logger.info("Adding a pairwise factor")

          factors += factor
          None
        }
      case c: Implication[_, _] =>
        val leftVariable = processConstraints(
          c.p.negate,
          instanceVariableMap, factorGraph, factors, variables, createVariable = true
        ).get
        val rightVariable = processConstraints(
          c.q,
          instanceVariableMap, factorGraph, factors, variables, createVariable = true
        ).get

        if (createVariable) {
          val outputVariable = getOutputVariable(factorGraph, state = true)
          val costs = SRMPFactorUtils.getPairDisjunctionCosts(leftVariable._2, rightVariable._2, Some(true))
          val factor = factorGraph.addFactor(3, Array(leftVariable._1, rightVariable._1, outputVariable), costs, FactorType.GeneralFactorType)

          //          logger.info("Adding a factor")

          factors += factor
          variables += outputVariable

          Some((outputVariable, true))
        } else {
          val costs = SRMPFactorUtils.getPairDisjunctionCosts(leftVariable._2, rightVariable._2)
          val factor = factorGraph.addPairwiseFactor(leftVariable._1, rightVariable._1, costs)

          //          logger.info("Adding a pairwise factor")

          factors += factor
          None
        }
      case c: Negation[_] =>
        //        val constraint = transformToLBJConstraint(c.p)
        logger.warn("Unsupported!")
        // Verify this once.
        //        new FirstOrderNegation(constraint)
        None
      case c: ForAll[_, _] =>
        logger.warn("Unsupported!")
        //        new FirstOrderConstant(true)
        None
      case c: AtLeast[_, _] =>
        logger.warn("Unsupported!")
        //        new FirstOrderConstant(true)
        None
      case c: AtMost[_, _] =>
        logger.warn("Unsupported!")
        //        new FirstOrderConstant(true)
        None
      case c: Exactly[_, _] =>
        logger.warn("Unsupported!")
        //        new FirstOrderConstant(true)
        None
      case c: ConstraintCollections[_, _] =>
        logger.warn("Unsupported!")
        //        new FirstOrderConstant(true)
        //        c.constraints.foldRight(Set[Any]()) {
        //          case (singleConstraint, ins) =>
        //            ins union getInstancesInvolved(singleConstraint).asInstanceOf[Set[Any]]
        //        }
        None
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
}
