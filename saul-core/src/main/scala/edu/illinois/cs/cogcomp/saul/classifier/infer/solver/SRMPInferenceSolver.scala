/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.solver

import edu.illinois.cs.cogcomp.lbjava.classify.{ Score, ScoreSet }
import edu.illinois.cs.cogcomp.lbjava.learn.{ Learner, Softmax }
import edu.illinois.cs.cogcomp.lbjava.infer.{ Constraint => LBJConstraint, _ }

import edu.illinois.cs.cogcomp.saul.classifier.infer.{ Assignment, Constraint }
import edu.illinois.cs.cogcomp.saul.classifier.infer.factorgraph.SRMPFactorUtils
import edu.illinois.cs.cogcomp.saul.util.Logging

import srmp.{ EnergyOptions, Factor, FactorType, Energy => FactorGraph, Node => FactorNode, SolverType => SRMPSolverType }

import scala.collection.mutable

final class SRMPInferenceSolver[T <: AnyRef, HEAD <: AnyRef] extends InferenceSolver[T, HEAD] with Logging {
  override def solve(constraintsOpt: Option[Constraint[_]], priorAssignment: Seq[Assignment]): Seq[Assignment] = {
    val softmax = new Softmax()
    val classifierLabelMap = new mutable.HashMap[Learner, List[(String, Int)]]()
    val instanceVariableMap = new mutable.HashMap[(Learner, String, Any), (FactorNode, Boolean)]()

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

      classifierLabelMap += (assignment.learner.classifier -> labelIndexMap)

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
                instanceVariableMap += ((assignment.learner.classifier, label, instance) -> (node, idx == 1))
            })
          } else {
            logger.info("Multi-variables thingy")
            val multiNodes = labelIndexMap.map({
              case (label: String, _) =>
                /* Normalized Probability Scores */
                val score = -1 * normalizedScoreset.getScore(label).score
                val node = factorGraph.addNode(2, Array(-1 - score, score))

                instanceVariableMap += ((assignment.learner.classifier, label, instance) -> (node, true))
                node
            })

            // XXX - Add an XOR factor for multiNodes
          }
      })
    })

    if (constraintsOpt.nonEmpty) {
      val lbjConstraints = LBJavaILPInferenceSolver.transformToLBJConstraint(constraintsOpt.get, new mutable.HashSet[FirstOrderVariable]())

      // Using the LBJava Constraint expressions to simplify First-Order Logic
      val propositional = {
        lbjConstraints match {
          case propositionalConjunction: PropositionalConjunction => propositionalConjunction.simplify(true)
          case _ => lbjConstraints.asInstanceOf[PropositionalConstraint].simplify()
        }
      }

      processConstraints(propositional, instanceVariableMap, factorGraph, factors, variables, isTopLevel = true)
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
        val domain = classifierLabelMap(assignment.learner.classifier)

        assignment.foreach({
          case (instance: Any, _) =>
            val newScores = domain.map({
              case (label: String, idx: Int) =>
                val variableTuple = instanceVariableMap((assignment.learner.classifier, label, instance))
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
    constraint: LBJConstraint,
    instanceVariableMap: mutable.HashMap[(Learner, String, Any), (FactorNode, Boolean)],
    factorGraph: FactorGraph,
    factors: mutable.ListBuffer[Factor],
    variables: mutable.ListBuffer[FactorNode],
    isTopLevel: Boolean
  ): Option[(FactorNode, Boolean)] = {
    constraint match {
      case c: PropositionalConstant =>
        // Nothing to do if the constraint is a constant
        // Ideally there should not be any constants
        logger.info("Processing PropositionalConstant - No Operation")
        None
      case c: PropositionalVariable =>
        logger.debug("Processing PropositionalVariable")
        val binaryVariable = instanceVariableMap((c.getClassifier, c.getPrediction, c.getExample))

        if (isTopLevel) {
          // XXX - Free Variables in the top-level conjunction are treated as grounded variables.
          // binaryVariable.setLogPotential(maxLogPotential)
        }

        // XXX - Verify this.
        Some(binaryVariable)
      case c: PropositionalConjunction =>
        logger.debug("Processing PropositionalConjunction")

        val variablesWithStates = c.getChildren.flatMap({ childConstraint: LBJConstraint =>
          // Should return variables with states if this is not a top-level conjunction.
          processConstraints(
            childConstraint,
            instanceVariableMap,
            factorGraph,
            factors,
            variables,
            isTopLevel = isTopLevel
          )
        })

        if (isTopLevel) {
          // Top-Level conjunction does not need to be enforced with a factor.
          None
        } else {
          //          logger.info("Non-toplevel conjunction")
          val firstPairOutput = variablesWithStates.head

          val outputVariableWithState = variablesWithStates.tail
            .foldLeft(firstPairOutput)({
              case (computedVariableWithState, currentVariableWithState) =>
                // XXX - This needs to be generalized.
                val intermediateVariable = getOutputVariable(factorGraph, state = true)
                val costs = SRMPFactorUtils.getPairConjunctionCosts(
                  computedVariableWithState._2,
                  currentVariableWithState._2,
                  Some(true)
                )

                val factor = factorGraph.addFactor(
                  3,
                  Array(computedVariableWithState._1, currentVariableWithState._1, intermediateVariable),
                  costs,
                  FactorType.GeneralFactorType
                )

                factors += factor
                variables += intermediateVariable

                (intermediateVariable, true)
            })

          Some(outputVariableWithState)
        }
      case c: PropositionalDisjunction =>
        logger.debug("Processing PropositionalDisjunction")
        //        if (createVariable) {
        //          val outputVariable = getOutputVariable(factorGraph, state = true)
        //          val costs = SRMPFactorUtils.getPairDisjunctionCosts(leftVariable._2, rightVariable._2, Some(true))
        //          val factor = factorGraph.addFactor(3, Array(leftVariable._1, rightVariable._1, outputVariable), costs, FactorType.GeneralFactorType)
        //
        //          //          logger.info("Adding a factor")
        //
        //          factors += factor
        //          variables += outputVariable
        //
        //          Some((outputVariable, true))
        //        } else {
        //          val costs = SRMPFactorUtils.getPairDisjunctionCosts(leftVariable._2, rightVariable._2)
        //          val factor = factorGraph.addPairwiseFactor(leftVariable._1, rightVariable._1, costs)
        //
        //          //          logger.info("Adding a pairwise factor")
        //
        //          factors += factor
        //          None
        //        }

        val variablesWithStates = c.getChildren.map({ childConstraint: LBJConstraint =>
          // Should return variables with states.
          processConstraints(
            childConstraint,
            instanceVariableMap,
            factorGraph,
            factors,
            variables,
            isTopLevel = false
          ).get
        })

        if (variablesWithStates.length == 1) {
          variablesWithStates.headOption
        } else {
          val firstPairOutput = variablesWithStates.head
          val outputVariableWithState = variablesWithStates.tail
            .foldLeft(firstPairOutput)({
              case (computedVariableWithState, currentVariableWithState) =>
                // XXX - This needs to be generalized.
                val intermediateVariable = getOutputVariable(factorGraph, state = true)
                val costs = SRMPFactorUtils.getPairDisjunctionCosts(
                  computedVariableWithState._2,
                  currentVariableWithState._2,
                  Some(true)
                )
                val factor = factorGraph.addFactor(
                  3,
                  Array(computedVariableWithState._1, currentVariableWithState._1, intermediateVariable),
                  costs,
                  FactorType.GeneralFactorType
                )

                factors += factor
                variables += intermediateVariable

                (intermediateVariable, true)
            })

          if (isTopLevel) {
            // XXX - Set the final output variable to true
          } else {
            // logger.info("Non-toplevel disjunction")
          }

          Some(outputVariableWithState)
        }

      case c: PropositionalNegation =>
        logger.debug("Processing PropositionalNegation")

        val childConstraints = c.getChildren
        if (childConstraints.length == 1 && childConstraints.head.isInstanceOf[PropositionalVariable]) {
          val childVariable = childConstraints.head.asInstanceOf[PropositionalVariable]
          val binaryVariable = instanceVariableMap((childVariable.getClassifier, childVariable.getPrediction, childVariable.getExample))

          if (isTopLevel) {
            // XXX - Free Variables in the top-level conjunction are treated as grounded variables.
            // binaryVariable.setLogPotential(-maxLogPotential)
          }

          // XXX - Verify this
          Some((binaryVariable._1, !binaryVariable._2))
        } else {
          logger.error("This constraint should already be processed")
          None
        }
      case c: PropositionalAtLeast =>
        logger.debug("Processing PropositionalAtLeast")

        //        if (isTopLevel) {
        val variablesWithStates = c.getChildren
          .map({ childConstraint: LBJConstraint =>
            processConstraints(
              childConstraint,
              instanceVariableMap,
              factorGraph,
              factors,
              variables,
              isTopLevel = false
            ).get
          })
        val totalConstraints = variablesWithStates.length
        println(totalConstraints)
        //        }

        logger.error("PropositionalAtLeast - Not supported yet.")

        //          // Budget factor evaluates as <= budget
        //          val factor = factorGraph.createFactorBUDGET(
        //            variablesWithStates.map(_._1),
        //            variablesWithStates.map(_._2),
        //            totalConstraints - c.getM,
        //            true
        //          )
        //
        //          factors += factor
        //        } else {
        //
        //        }
        None
      case c: PropositionalImplication =>
        logger.info("Processing PropositionalImplication")
        logger.error("PropositionalImplication - This constraint should already be processed")
        None
      case c: PropositionalDoubleImplication =>
        logger.info("Processing PropositionalDoubleImplication")
        logger.error("PropositionalDoubleImplication - This constraint should already be processed")
        None
      case _ =>
        throw new Exception("Unknown constraint exception! This constraint should have been rewritten in terms of other constraints. ")
    }
  }
}
