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
import edu.illinois.cs.cogcomp.lbjava.infer.{ Constraint => LBJConstraint, PropositionalConstraint => LBJPropositionalConstraint, _ }
import edu.illinois.cs.cogcomp.lbjava.learn.{ Learner, Softmax }
import edu.illinois.cs.cogcomp.saul.classifier.infer.{ Assignment, Constraint }
import edu.illinois.cs.cogcomp.saul.classifier.infer.factorgraph.{ BinaryRandomVariable, FactorUtils }
import edu.illinois.cs.cogcomp.saul.util.Logging

import scala.collection.mutable

final class MaxBPInferenceSolver[T <: AnyRef, HEAD <: AnyRef] extends InferenceSolver[T, HEAD] with Logging {
  override def solve(constraintsOpt: Option[Constraint[_]], priorAssignment: Seq[Assignment]): Seq[Assignment] = {
    val softmax = new Softmax()
    val classifierLabelMap = new mutable.HashMap[Learner, List[(String, Boolean)]]()
    val instanceVariableMap = new mutable.HashMap[(Learner, String, Any), (BinaryRandomVariable, Boolean)]()

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

      classifierLabelMap += (assignment.learner.classifier -> labelIndexMap)

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
                instanceVariableMap += ((assignment.learner.classifier, label, instance) -> (binaryVariable, state))
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
              instanceVariableMap += ((assignment.learner.classifier, label, instance) -> (binaryVariable, true))

              factors ++= family.Factor(binaryVariable)
              variables += binaryVariable

              binaryVariable
            })

            // XXX - binaryVariables needs an ExactlyOne factor
            logger.error("Convert to XOR - Unsuppported!")
          }
      })
    })

    if (constraintsOpt.nonEmpty) {
      val lbjConstraints = LBJavaILPInferenceSolver.transformToLBJConstraint(constraintsOpt.get, new mutable.HashSet[FirstOrderVariable]())

      // Using the LBJava Constraint expressions to simplify First-Order Logic
      val propositional = {
        lbjConstraints match {
          case propositionalConjunction: PropositionalConjunction => propositionalConjunction.simplify(true)
          case _ => lbjConstraints.asInstanceOf[LBJPropositionalConstraint].simplify()
        }
      }

      processConstraints(propositional, instanceVariableMap, factors, variables, isTopLevel = true)
    }

    val model = new ItemizedModel(factors)
    val loopyMaxSummary = LoopyBPSummaryMaxProduct(variables, BPMaxProductRing, model)
    BP.inferLoopyMax(loopyMaxSummary)

    val assignment = loopyMaxSummary.maximizingAssignment
    val fg = new MAPSummary(assignment, loopyMaxSummary.factors.get.toVector)

    if (factors.exists(_.assignmentScore(fg.mapAssignment) == Double.NegativeInfinity)) {
      logger.error("Unsatisfied Factors exist. Using local assignments!")
      return priorAssignment
    }

    val finalAssignments = priorAssignment.map({ assignment: Assignment =>
      val finalAssgn = Assignment(assignment.learner)
      val domain = classifierLabelMap(assignment.learner.classifier)

      assignment.foreach({
        case (instance: Any, _) =>
          val newScores = domain.map({
            case (label: String, _) =>
              val variable = instanceVariableMap((assignment.learner.classifier, label, instance))
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
    constraint: LBJPropositionalConstraint,
    instanceVariableMap: mutable.HashMap[(Learner, String, Any), (BinaryRandomVariable, Boolean)],
    factors: mutable.ListBuffer[Factor],
    variables: mutable.ListBuffer[BinaryRandomVariable],
    isTopLevel: Boolean
  ): Option[(BinaryRandomVariable, Boolean)] = {
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
          val unaryFactor = FactorUtils.getUnaryFactor(binaryVariable)
          factors += unaryFactor
        }

        // XXX - Verify this.
        Some(binaryVariable)
      case c: PropositionalConjunction =>
        logger.debug("Processing PropositionalConjunction")
        val variablesWithStates = c.getChildren.flatMap({ childConstraint: LBJConstraint =>
          // Should return variables with states if this is not a top-level conjunction.
          processConstraints(
            childConstraint.asInstanceOf[LBJPropositionalConstraint],
            instanceVariableMap,
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
                val intermediateVariable = new BinaryRandomVariable(true)
                val factor = FactorUtils.getPairConjunctionFactor(
                  computedVariableWithState,
                  currentVariableWithState,
                  Some((intermediateVariable, true))
                )

                factors += factor
                variables += intermediateVariable

                (intermediateVariable, true)
            })

          Some(outputVariableWithState)
        }
      case c: PropositionalDisjunction =>
        logger.debug("Processing PropositionalDisjunction")
        val variablesWithStates = c.getChildren.map({ childConstraint: LBJConstraint =>
          // Should return variables with states.
          processConstraints(
            childConstraint.asInstanceOf[LBJPropositionalConstraint],
            instanceVariableMap,
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
                val intermediateVariable = new BinaryRandomVariable(true)
                val factor = FactorUtils.getPairDisjunctionFactor(
                  computedVariableWithState,
                  currentVariableWithState,
                  Some((intermediateVariable, true))
                )

                factors += factor
                variables += intermediateVariable

                (intermediateVariable, true)
            })

          if (isTopLevel) {
            // Set the final output variable to true
            val unaryFactor = FactorUtils.getUnaryFactor(outputVariableWithState)
            factors += unaryFactor
          } else {
            //             logger.info("Non-toplevel disjunction")
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
            // Free Variables in the top-level conjunction are treated as grounded variables.
            val unaryFactor = FactorUtils.getUnaryFactor((binaryVariable._1, !binaryVariable._2))
            factors += unaryFactor
          }

          // XXX - Verify this
          Some((binaryVariable._1, !binaryVariable._2))
        } else {
          logger.error("This constraint should already be processed")
          None
        }
      case c: PropositionalAtLeast =>
        logger.debug("Processing PropositionalAtLeast")
        logger.info(s"Atleast ${c.getM} of ${c.size()} - $isTopLevel")

        if (c.getM == 1) {
          // Treat this as a disjunction.
          val childConstraints = c.getChildren
          if (childConstraints.length == 1) {
            processConstraints(
              childConstraints.head.asInstanceOf[LBJPropositionalConstraint],
              instanceVariableMap,
              factors,
              variables,
              isTopLevel = isTopLevel
            )
          } else {
            val disjunction = new PropositionalDisjunction(
              childConstraints.head.asInstanceOf[LBJPropositionalConstraint],
              childConstraints(1).asInstanceOf[LBJPropositionalConstraint]
            )

            childConstraints.drop(2).foreach({ constraint: LBJConstraint =>
              disjunction.add(constraint.asInstanceOf[LBJPropositionalConstraint])
            })

            processConstraints(
              disjunction,
              instanceVariableMap,
              factors,
              variables,
              isTopLevel = isTopLevel
            )
          }
        } else if (c.getM == c.size() - 1) {
          // Treat this as a disjunction of conjunctions
          val childConstraints = c.getChildren
          val innerConjunctions = childConstraints.combinations(c.getM)
            .map(childSubset => {
              if (childSubset.length == 1) {
                childSubset.head.asInstanceOf[LBJPropositionalConstraint]
              } else {
                val conjunction = new PropositionalConjunction(
                  childSubset.head.asInstanceOf[LBJPropositionalConstraint],
                  childSubset(1).asInstanceOf[LBJPropositionalConstraint]
                )

                childSubset.drop(2).foreach(c => conjunction.add(c.asInstanceOf[LBJPropositionalConstraint]))

                conjunction
              }
            }).toArray

          if (innerConjunctions.length == 1) {
            processConstraints(
              innerConjunctions.head,
              instanceVariableMap,
              factors,
              variables,
              isTopLevel = isTopLevel
            )
          } else {
            val outerDisjunction = new PropositionalDisjunction(
              innerConjunctions.head,
              innerConjunctions(1)
            )
            innerConjunctions.drop(2).foreach(c => outerDisjunction.add(c))
            processConstraints(
              innerConjunctions.head,
              instanceVariableMap,
              factors,
              variables,
              isTopLevel = isTopLevel
            )
          }

        } else {
          //          val variablesWithStates = c.getChildren
          //            .map({ childConstraint: LBJConstraint =>
          //              processConstraints(
          //                childConstraint,
          //                instanceVariableMap,
          //                factors,
          //                variables,
          //                isTopLevel = false
          //              ).get
          //            })
          //
          //          val totalConstraints = variablesWithStates.length

          // XXX - Not implemented yet.
          logger.error("PropositionalAtLeast - Not supported yet.")
        }

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
