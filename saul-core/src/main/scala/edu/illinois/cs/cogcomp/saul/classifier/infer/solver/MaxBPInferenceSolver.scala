/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.solver

import cc.factorie.DenseTensor1
import cc.factorie.infer._
import cc.factorie.model.{ DotTemplateWithStatistics1, Factor, ItemizedModel, Parameters }
import edu.illinois.cs.cogcomp.lbjava.classify.{ Score, ScoreSet }
import edu.illinois.cs.cogcomp.lbjava.learn.Softmax
import edu.illinois.cs.cogcomp.saul.classifier.infer._
import edu.illinois.cs.cogcomp.saul.classifier.infer.factorgraph.{ BinaryRandomVariable, FactorUtils }
import edu.illinois.cs.cogcomp.saul.lbjrelated.LBJLearnerEquivalent
import edu.illinois.cs.cogcomp.saul.util.Logging

import scala.collection.mutable

final class MaxBPInferenceSolver[T <: AnyRef, HEAD <: AnyRef] extends InferenceSolver[T, HEAD] with Logging {
  override def solve(constraintsOpt: Option[Constraint[_]], priorAssignment: Seq[Assignment]): Seq[Assignment] = {
    val softmax = new Softmax()
    val classifierLabelMap = new mutable.HashMap[LBJLearnerEquivalent, List[(String, Boolean)]]()
    val instanceVariableMap = new mutable.HashMap[(LBJLearnerEquivalent, String, Any), (BinaryRandomVariable, Boolean)]()

    val factors = new mutable.ListBuffer[Factor]()
    val variables = new mutable.ListBuffer[BinaryRandomVariable]()

    priorAssignment.foreach({ assignment: Assignment =>
      val labels: List[String] = assignment.learner.classifier.scores(assignment.head._1).toArray.map(_.value).toList

      val family = new DotTemplateWithStatistics1[BinaryRandomVariable] with Parameters {
        val weights = Weights(new DenseTensor1(2))
      }

      val labelIndexMap = if (labels.size == 2) {
        labels.zip(Array(false, true))
      } else {
        labels.zip(Array.fill(labels.size)(true))
      }

      classifierLabelMap += (assignment.learner -> labelIndexMap)

      assignment.foreach({
        case (instance: Any, scores: ScoreSet) =>
          val normalizedScoreset = softmax.normalize(scores)

          if (labels.size == 2) {
            val highScoreValue = normalizedScoreset.highScoreValue()
            val defaultIdx = labelIndexMap.filter(_._1 == highScoreValue).map(_._2).head
            val binaryVariable = new BinaryRandomVariable(defaultIdx, classifier = assignment.learner.classifier.name)

            labelIndexMap.foreach({
              case (label: String, state: Boolean) =>
                val idx = if (state) 1 else 0
                family.weights.value(idx) = math.log(normalizedScoreset.getScore(label).score)
                instanceVariableMap += ((assignment.learner, label, instance) -> (binaryVariable, state))
            })

            factors += family.Factor(binaryVariable)
            variables += binaryVariable
          } else {
            val binaryVariables = labels.map({ label: String =>

              /* Normalized Probability Scores */
              val score = normalizedScoreset.getScore(label).score
              family.weights.value(1) = math.log(score)
              family.weights.value(0) = math.log(1 - score)

              val binaryVariable = new BinaryRandomVariable(score >= 0.5, classifier = s"${assignment.learner.classifier.name}_$label")
              instanceVariableMap += ((assignment.learner, label, instance) -> (binaryVariable, true))

              factors ++= family.Factor(binaryVariable)
              variables += binaryVariable

              binaryVariable
            })

            //          binaryVariables needs an ExactlyOne factor
            logger.warn("Unsuppported!")
          }
      })
    })

    if (constraintsOpt.nonEmpty)
      processConstraints(constraintsOpt.get, instanceVariableMap, factors, variables)

    val model = new ItemizedModel(factors)
    val loopyMaxSummary = LoopyBPSummaryMaxProduct(variables, BPMaxProductRing, model)
    BP.inferLoopyMax(loopyMaxSummary)

    val assignment = loopyMaxSummary.maximizingAssignment
    val fg = new MAPSummary(assignment, loopyMaxSummary.factors.get.toVector)

    if (factors.exists(_.assignmentScore(fg.mapAssignment) == Double.NegativeInfinity)) {
      println("Unsatisfied Factors exist.")
    }

    val finalAssignments = priorAssignment.map({ assignment: Assignment =>
      val finalAssgn = Assignment(assignment.learner)
      val domain = classifierLabelMap(assignment.learner)

      assignment.foreach({
        case (instance: Any, _) =>
          val newScores = domain.map({
            case (label: String, _) =>
              val variable = instanceVariableMap((assignment.learner, label, instance))
              val mapAssignment = if (fg.mapAssignment(variable._1).category == variable._2) 1.0 else 0.0
              new Score(label, mapAssignment)
          }).toArray

          finalAssgn += (instance -> new ScoreSet(newScores))
      })

      finalAssgn
    })

    finalAssignments
  }

  private def processConstraints(
    constraint: Constraint[_],
    instanceVariableMap: mutable.HashMap[(LBJLearnerEquivalent, String, Any), (BinaryRandomVariable, Boolean)],
    factors: mutable.ListBuffer[Factor],
    variables: mutable.ListBuffer[BinaryRandomVariable],
    createVariable: Boolean = false
  ): Option[(BinaryRandomVariable, Boolean)] = {
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
          instanceVariableMap, factors, variables, createVariable = true
        ).get
        val rightVariable = processConstraints(
          c.c2,
          instanceVariableMap, factors, variables, createVariable = true
        ).get

        if (createVariable) {
          val outputVariable = new BinaryRandomVariable(true)

          // Handle negation etc.
          val factor = FactorUtils.getPairConjunctionFactor(leftVariable, rightVariable, Some((outputVariable, true)))

          factors += factor
          variables += outputVariable

          Some((outputVariable, true))
        } else {
          // Handle negation etc.
          val factor = FactorUtils.getPairConjunctionFactor(leftVariable, rightVariable)
          factors += factor
          None
        }
      case c: PairDisjunction[_, _] =>
        val leftVariable = processConstraints(
          c.c1,
          instanceVariableMap, factors, variables, createVariable = true
        ).get
        val rightVariable = processConstraints(
          c.c2,
          instanceVariableMap, factors, variables, createVariable = true
        ).get

        if (createVariable) {
          val outputVariable = new BinaryRandomVariable(true)

          // Handle negation etc.
          val factor = FactorUtils.getPairDisjunctionFactor(leftVariable, rightVariable, Some((outputVariable, true)))

          factors += factor
          variables += outputVariable

          Some((outputVariable, true))
        } else {
          // Handle negation etc.
          val factor = FactorUtils.getPairDisjunctionFactor(leftVariable, rightVariable)
          factors += factor
          None
        }
      case c: Implication[_, _] =>
        val leftVariable = processConstraints(
          c.p.negate,
          instanceVariableMap, factors, variables, createVariable = true
        ).get
        val rightVariable = processConstraints(
          c.q,
          instanceVariableMap, factors, variables, createVariable = true
        ).get

        if (createVariable) {
          val outputVariable = new BinaryRandomVariable(true)

          // Handle negation etc.
          val factor = FactorUtils.getPairDisjunctionFactor(leftVariable, rightVariable, Some(outputVariable, true))

          factors += factor
          variables += outputVariable

          Some((outputVariable, true))
        } else {
          // Handle negation etc.
          val factor = FactorUtils.getPairDisjunctionFactor(leftVariable, rightVariable)
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
