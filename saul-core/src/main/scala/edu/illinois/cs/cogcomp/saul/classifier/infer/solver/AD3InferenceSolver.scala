/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.solver

import edu.cmu.cs.ark.ad3.{ BinaryVariable, Factor, FactorGraph, MAPResult }
import edu.illinois.cs.cogcomp.lbjava.classify.{ Score, ScoreSet }
import edu.illinois.cs.cogcomp.lbjava.learn.Softmax
import edu.illinois.cs.cogcomp.saul.classifier.infer._
import edu.illinois.cs.cogcomp.saul.lbjrelated.LBJLearnerEquivalent
import edu.illinois.cs.cogcomp.saul.util.Logging

import scala.collection.mutable

final class AD3InferenceSolver[T <: AnyRef, HEAD <: AnyRef] extends InferenceSolver[T, HEAD] with Logging {
  private val maxLogPotential = 100

  override def solve(constraintsOpt: Option[Constraint[_]], priorAssignment: Seq[Assignment]): Seq[Assignment] = {
    val softmax = new Softmax()
    val classifierLabelMap = new mutable.HashMap[LBJLearnerEquivalent, List[String]]()
    val instanceVariableMap = new mutable.HashMap[(LBJLearnerEquivalent, String, Any), (BinaryVariable, Boolean)]()

    val factorGraph = new FactorGraph()

    val factors = new mutable.ListBuffer[Factor]()
    val variables = new mutable.ListBuffer[BinaryVariable]()

    priorAssignment.foreach({ assignment: Assignment =>
      val labels: List[String] = assignment.learner.classifier.scores(assignment.head._1).toArray.map(_.value).toList.sorted
      classifierLabelMap += (assignment.learner -> labels)

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

              instanceVariableMap += ((assignment.learner, label, instance) -> (binaryVariable, true))
          })
      })
    })

    if (constraintsOpt.nonEmpty)
      processConstraints(constraintsOpt.get, instanceVariableMap, classifierLabelMap, factorGraph, factors, variables)

    factorGraph.setVerbosity(0)
    factorGraph.fixMultiVariablesWithoutFactors()

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
        val domain = classifierLabelMap(assignment.learner)

        assignment.foreach({
          case (instance: Any, _) =>
            val newScores = domain.map({
              case (label: String) =>
                val variableTuple = instanceVariableMap((assignment.learner, label, instance))
                val binaryVariable = variableTuple._1
                val state = variableTuple._2
                val posterior = mapResult.variablePosteriors(binaryVariable.getId)

                new Score(label, if (state) posterior else 1.0 - posterior)
            }).toArray

            finalAssgn += (instance -> new ScoreSet(newScores))
        })

        finalAssgn
      })

      finalAssignments
    }
  }

  private def processConstraints(
    constraint: Constraint[_],
    instanceVariableMap: mutable.HashMap[(LBJLearnerEquivalent, String, Any), (BinaryVariable, Boolean)],
    classifierLabelMap: mutable.HashMap[LBJLearnerEquivalent, List[String]],
    factorGraph: FactorGraph,
    factors: mutable.ListBuffer[Factor],
    variables: mutable.ListBuffer[BinaryVariable],
    createVariable: Boolean = false
  ): Option[(BinaryVariable, Boolean)] = {
    constraint match {
      case c: PropositionalEqualityConstraint[_] =>
        val value = c.equalityValOpt.orElse(c.inequalityValOpt).get
        val variableWithState = instanceVariableMap((c.estimator, value, c.instanceOpt.get))

        // Handle negative variable states
        val isEquality = c.equalityValOpt.nonEmpty == variableWithState._2

        if (createVariable) {
          Some((variableWithState._1, isEquality))
        } else {
          // If this constraint is not a top-level constraint, add a factor to force its value.
          val outputVariable = factorGraph.createBinaryVariable()
          outputVariable.setLogPotential(maxLogPotential)

          val factor = factorGraph.createFactorIMPLY(
            Array(outputVariable, variableWithState._1),
            Array(false, !isEquality),
            true
          )
          factors += factor
          variables += outputVariable

          None
        }

      case c: InstancePairEqualityConstraint[_] =>
        val isEquality = c.equalsOpt.get
        val labels = classifierLabelMap(c.estimator)
        val firstVariables = labels.map({ label =>
          instanceVariableMap((c.estimator, label, c.instance1))
        })
        val secondVariables = labels.map({ label =>
          instanceVariableMap((c.estimator, label, c.instance2Opt.get))
        })

        if (createVariable) {
          val outputVariable = factorGraph.createBinaryVariable()
          outputVariable.setLogPotential(maxLogPotential)

          firstVariables.zip(secondVariables)
            .foreach({
              case ((firstVar: BinaryVariable, firstState: Boolean), (secondVar: BinaryVariable, secondState: Boolean)) =>
                val firstNegatedState = if (isEquality) firstState else !firstState
                val secondNegatedState = !secondState
                factors += factorGraph.createFactorXOROUT(
                  Array(firstVar, secondVar, outputVariable),
                  Array(firstNegatedState, secondNegatedState, false),
                  true
                )
            })

          variables += outputVariable

          Some((outputVariable, true))
        } else {
          firstVariables.zip(secondVariables)
            .foreach({
              case ((firstVar: BinaryVariable, firstState: Boolean), (secondVar: BinaryVariable, secondState: Boolean)) =>
                val firstNegatedState = if (isEquality) firstState else !firstState
                val secondNegatedState = !secondState
                factors += factorGraph.createFactorXOR(
                  Array(firstVar, secondVar),
                  Array(firstNegatedState, secondNegatedState),
                  true
                )
            })

          None
        }

      //      case c: EstimatorPairEqualityConstraint[_] =>

      case c: PairConjunction[_, _] =>
        val leftVariable = processConstraints(
          c.c1,
          instanceVariableMap, classifierLabelMap, factorGraph, factors, variables, createVariable = true
        ).get
        val rightVariable = processConstraints(
          c.c2,
          instanceVariableMap, classifierLabelMap, factorGraph, factors, variables, createVariable = true
        ).get

        if (createVariable) {
          val outputVariable = factorGraph.createBinaryVariable()
          outputVariable.setLogPotential(maxLogPotential)

          val factor = factorGraph.createFactorANDOUT(
            Array(leftVariable._1, rightVariable._1, outputVariable),
            Array(!leftVariable._2, !rightVariable._2, false),
            true
          )

          factors += factor
          variables += outputVariable

          Some((outputVariable, true))
        } else {
          val outputVariable = factorGraph.createBinaryVariable()
          outputVariable.setLogPotential(maxLogPotential)

          val factor = factorGraph.createFactorANDOUT(
            Array(leftVariable._1, rightVariable._1, outputVariable),
            Array(!leftVariable._2, !rightVariable._2, false),
            true
          )

          factors += factor
          None
        }
      case c: PairDisjunction[_, _] =>
        val leftVariable = processConstraints(
          c.c1,
          instanceVariableMap, classifierLabelMap, factorGraph, factors, variables, createVariable = true
        ).get
        val rightVariable = processConstraints(
          c.c2,
          instanceVariableMap, classifierLabelMap, factorGraph, factors, variables, createVariable = true
        ).get

        if (createVariable) {
          val outputVariable = factorGraph.createBinaryVariable()
          outputVariable.setLogPotential(maxLogPotential)

          // Handle negation etc.
          val factor = factorGraph.createFactorOROUT(
            Array(leftVariable._1, rightVariable._1, outputVariable),
            Array(!leftVariable._2, !rightVariable._2, false),
            true
          )

          factors += factor
          variables += outputVariable

          Some((outputVariable, true))
        } else {
          // Handle negation etc.
          val factor = factorGraph.createFactorOR(
            Array(leftVariable._1, rightVariable._1),
            Array(!leftVariable._2, !rightVariable._2),
            true
          )

          factors += factor
          None
        }
      case c: Implication[_, _] =>
        val leftVariable = processConstraints(
          c.p.negate,
          instanceVariableMap, classifierLabelMap, factorGraph, factors, variables, createVariable = true
        ).get
        val rightVariable = processConstraints(
          c.q,
          instanceVariableMap, classifierLabelMap, factorGraph, factors, variables, createVariable = true
        ).get

        if (createVariable) {
          val outputVariable = factorGraph.createBinaryVariable()
          outputVariable.setLogPotential(maxLogPotential)

          val factor = factorGraph.createFactorOROUT(
            Array(leftVariable._1, rightVariable._1, outputVariable),
            Array(!leftVariable._2, !rightVariable._2, false),
            true
          )

          factors += factor
          variables += outputVariable

          Some((outputVariable, true))
        } else {
          val factor = factorGraph.createFactorOR(
            Array(leftVariable._1, rightVariable._1),
            Array(!leftVariable._2, !rightVariable._2),
            true
          )

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
        // ForAll is a conjunction
        val processedConstraints = c.constraints.map({
          cons =>
            processConstraints(cons, instanceVariableMap, classifierLabelMap, factorGraph, factors, variables, createVariable = true).get
        })

        val binaryVariables = processedConstraints.map(_._1).toArray
        val isNegated = processedConstraints.toArray.map(ins => !ins._2)

        val outputVariable = factorGraph.createBinaryVariable()
        outputVariable.setLogPotential(maxLogPotential)

        val factor = factorGraph.createFactorANDOUT(
          binaryVariables :+ outputVariable,
          isNegated :+ false,
          true
        )

        factors += factor
        variables += outputVariable

        if (createVariable) {
          Some((outputVariable, true))
        } else {
          None
        }
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
