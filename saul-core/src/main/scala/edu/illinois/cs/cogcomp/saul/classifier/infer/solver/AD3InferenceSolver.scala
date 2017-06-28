/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.solver

import edu.cmu.cs.ark.ad3.{ BinaryVariable, Factor, FactorGraph, MAPResult }
import edu.illinois.cs.cogcomp.lbjava.classify.{ Score, ScoreSet }
import edu.illinois.cs.cogcomp.lbjava.infer.{ Constraint => LBJConstraint, _ }
import edu.illinois.cs.cogcomp.lbjava.learn.{ Learner, Softmax }
import edu.illinois.cs.cogcomp.saul.classifier.infer.{ Assignment, Constraint }
import edu.illinois.cs.cogcomp.saul.util.Logging

import scala.collection.mutable

final class AD3InferenceSolver[T <: AnyRef, HEAD <: AnyRef] extends InferenceSolver[T, HEAD] with Logging {
  private val maxLogPotential = 100

  override def solve(constraintsOpt: Option[Constraint[_]], priorAssignment: Seq[Assignment]): Seq[Assignment] = {
    val softmax = new Softmax()
    val classifierLabelMap = new mutable.HashMap[Learner, List[String]]()
    val instanceVariableMap = new mutable.HashMap[(Learner, String, Any), BinaryVariable]()

    val factorGraph = new FactorGraph()

    val factors = new mutable.ListBuffer[Factor]()
    val variables = new mutable.ListBuffer[BinaryVariable]()

    priorAssignment.foreach({ assignment: Assignment =>
      val labels: List[String] = assignment.learner.classifier.scores(assignment.head._1).toArray.map(_.value).toList.sorted
      classifierLabelMap += (assignment.learner.classifier -> labels)

      assignment.foreach({
        case (instance: Any, scores: ScoreSet) =>
          val normalizedScoreset = softmax.normalize(scores)

          val multiVariable = factorGraph.createMultiVariable(labels.size)
          labels.zipWithIndex.map({
            case (label: String, idx: Int) =>

              /* Normalized Probability Scores */
              val score = normalizedScoreset.getScore(label).score
              multiVariable.setLogPotential(idx, score)

              val binaryVariable = multiVariable.getState(idx)
              variables += binaryVariable

              instanceVariableMap += ((assignment.learner.classifier, label, instance) -> binaryVariable)
          })
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

      processConstraints(
        propositional,
        instanceVariableMap, classifierLabelMap, factorGraph, factors, variables, isTopLevel = true
      )
    }

    factorGraph.setVerbosity(0)
    factorGraph.fixMultiVariablesWithoutFactors()

    // Solve Exact MAP problem using AD3
    val mapResult: MAPResult = factorGraph.solveExactMAPWithAD3()

    if (mapResult.status == 2) {
      logger.warn("Infeasible problem. Using original assignments.")
      priorAssignment
    } else if (mapResult.status == 3) {
      logger.warn("Unsolved problem. Using original assignment.")
      priorAssignment
    } else {
      val finalAssignments = priorAssignment.map({ assignment: Assignment =>
        val finalAssgn = Assignment(assignment.learner)
        val domain = classifierLabelMap(assignment.learner.classifier)

        assignment.foreach({
          case (instance: Any, _) =>
            val newScores = domain.map({
              case (label: String) =>
                val binaryVariable = instanceVariableMap((assignment.learner.classifier, label, instance))
                val posterior = mapResult.variablePosteriors(binaryVariable.getId)

                new Score(label, posterior)
            }).toArray

            finalAssgn += (instance -> new ScoreSet(newScores))
        })

        finalAssgn
      })

      finalAssignments
    }
  }

  private def processConstraints(
    constraint: LBJConstraint,
    instanceVariableMap: mutable.HashMap[(Learner, String, Any), BinaryVariable],
    classifierLabelMap: mutable.HashMap[Learner, List[String]],
    factorGraph: FactorGraph,
    factors: mutable.ListBuffer[Factor],
    variables: mutable.ListBuffer[BinaryVariable],
    isTopLevel: Boolean
  ): Option[(BinaryVariable, Boolean)] = {
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
          // Free Variables in the top-level conjunction are treated as grounded variables.
          binaryVariable.setLogPotential(maxLogPotential)
        }

        Some((binaryVariable, true))
      case c: PropositionalConjunction =>
        logger.debug("Processing PropositionalConjunction")
        val variablesWithStates = c.getChildren.flatMap({ childConstraint: LBJConstraint =>
          // Should return variables with states if this is not a top-level conjunction.
          processConstraints(
            childConstraint,
            instanceVariableMap,
            classifierLabelMap,
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
          val outputVariable = factorGraph.createBinaryVariable()
          val factor = factorGraph.createFactorANDOUT(
            variablesWithStates.map(_._1) :+ outputVariable,
            variablesWithStates.map(!_._2) :+ false,
            true
          )

          factors += factor
          variables += outputVariable

          Some((outputVariable, true))
        }
      case c: PropositionalDisjunction =>
        logger.debug("Processing PropositionalDisjunction")
        val variablesWithStates = c.getChildren.map({ childConstraint: LBJConstraint =>
          // Should return variables with states.
          processConstraints(
            childConstraint,
            instanceVariableMap,
            classifierLabelMap,
            factorGraph,
            factors,
            variables,
            isTopLevel = false
          ).get
        })

        if (isTopLevel) {
          val factor = factorGraph.createFactorOR(
            variablesWithStates.map(_._1),
            variablesWithStates.map(!_._2),
            true
          )

          factors += factor
          None
        } else {
          //          logger.info("Non-toplevel disjunction")
          val outputVariable = factorGraph.createBinaryVariable()

          val factor = factorGraph.createFactorOROUT(
            variablesWithStates.map(_._1) :+ outputVariable,
            variablesWithStates.map(!_._2) :+ false,
            true
          )

          factors += factor
          variables += outputVariable

          Some((outputVariable, true))
        }

      case c: PropositionalNegation =>
        logger.debug("Processing PropositionalNegation")

        val childConstraints = c.getChildren
        if (childConstraints.length == 1 && childConstraints.head.isInstanceOf[PropositionalVariable]) {
          val childVariable = childConstraints.head.asInstanceOf[PropositionalVariable]
          val binaryVariable = instanceVariableMap((childVariable.getClassifier, childVariable.getPrediction, childVariable.getExample))

          if (isTopLevel) {
            // Free Variables in the top-level conjunction are treated as grounded variables.
            binaryVariable.setLogPotential(-maxLogPotential)
          }

          Some((binaryVariable, false))
        } else {
          logger.error("This constraint should already be processed")
          None
        }
      case c: PropositionalAtLeast =>
        logger.debug("Processing PropositionalAtLeast")
        if (isTopLevel) {
          val variablesWithStates = c.getChildren
            .map({ childConstraint: LBJConstraint =>
              processConstraints(
                childConstraint,
                instanceVariableMap,
                classifierLabelMap,
                factorGraph,
                factors,
                variables,
                isTopLevel = false
              ).get
            })

          val totalConstraints = variablesWithStates.length

          // Budget factor evaluates as <= budget
          val factor = factorGraph.createFactorBUDGET(
            variablesWithStates.map(_._1),
            variablesWithStates.map(_._2),
            totalConstraints - c.getM,
            true
          )

          factors += factor
        } else {
          logger.error("PropositionalAtLeast - Not supported yet.")
        }
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
