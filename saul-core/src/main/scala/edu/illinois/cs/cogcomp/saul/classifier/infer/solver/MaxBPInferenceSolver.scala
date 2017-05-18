/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.solver

import cc.factorie.DenseTensor1
import cc.factorie.Factorie.CategoricalVariable
import cc.factorie.infer.{ BP, BPMaxProductRing, BPSummary }
import cc.factorie.model.{ DotTemplateWithStatistics1, Factor, ItemizedModel, Parameters }
import cc.factorie.variable.{ CategoricalDomain, CategoricalValue }
import edu.illinois.cs.cogcomp.lbjava.classify.{ Score, ScoreSet }
import edu.illinois.cs.cogcomp.lbjava.learn.Softmax
import edu.illinois.cs.cogcomp.saul.classifier.infer.{ Assignment, Constraint }
import edu.illinois.cs.cogcomp.saul.lbjrelated.LBJLearnerEquivalent
import edu.illinois.cs.cogcomp.saul.util.Logging

import scala.collection.mutable

class MaxBPInferenceSolver[T <: AnyRef, HEAD <: AnyRef] extends InferenceSolver[T, HEAD] with Logging {
  override def solve(constraintsOpt: Option[Constraint[_]], priorAssignment: Seq[Assignment]): Seq[Assignment] = {
    val softmax = new Softmax()
    val classifierDomainMap = new mutable.HashMap[LBJLearnerEquivalent, CategoricalDomain[String]]()
    val instanceVariableMap = new mutable.HashMap[(LBJLearnerEquivalent, Any), CategoricalVariable[String]]()

    val factors = new mutable.ListBuffer[Factor]()
    val variables = new mutable.HashSet[CategoricalVariable[String]]()

    priorAssignment.foreach({ assignment: Assignment =>
      val labels: List[String] = assignment.learner.classifier.scores(assignment.head._1).toArray.map(_.value).toList

      object ClassifierDomain extends CategoricalDomain[String](labels)
      class ClassifierVariable(label: String, val name: String) extends CategoricalVariable[String] {
        override def domain: CategoricalDomain[String] = ClassifierDomain
      }

      val family = new DotTemplateWithStatistics1[ClassifierVariable] with Parameters {
        val weights = Weights(new DenseTensor1(ClassifierDomain.size))
      }

      classifierDomainMap += (assignment.learner -> ClassifierDomain)

      assignment.foreach({
        case (instance: Any, scores: ScoreSet) =>
          val normalizedScoreset = softmax.normalize(scores)
          ClassifierDomain.foreach({ domainItem: CategoricalValue[String] =>
            val score = math.log(normalizedScoreset.getScore(domainItem.category).score)
            family.weights.value(domainItem.intValue) = score
          })

          val variable = new ClassifierVariable(scores.highScoreValue(), "Testing")
          factors ++= family.factors(variable)
          variables += variable

          instanceVariableMap += ((assignment.learner, instance) -> variable)
      })
    })

    if (constraintsOpt.nonEmpty)
      processConstraints(constraintsOpt.get, classifierDomainMap, instanceVariableMap, factors, variables)

    val model = new ItemizedModel(factors)
    val fg = BPSummary(variables, BPMaxProductRing, model)
    BP.inferLoopyMax(fg)

    val finalAssignments = priorAssignment.map({ assignment: Assignment =>
      val finalAssgn = Assignment(assignment.learner)
      val domain = classifierDomainMap(assignment.learner)

      assignment.foreach({
        case (instance: Any, _) =>
          val variable = instanceVariableMap((assignment.learner, instance))
          val proportions = fg.marginal(variable).proportions

          val newScores = domain.map({ value: CategoricalValue[String] =>
            val index = value.singleIndex
            new Score(value.category, proportions(index))
          }).toArray

          finalAssgn += (instance -> new ScoreSet(newScores))
      })

      finalAssgn
    })

    finalAssignments
  }

  private def processConstraints(
    constraints: Constraint[_],
    classifierDomainMap: mutable.HashMap[LBJLearnerEquivalent, CategoricalDomain[String]],
    instanceVariableMap: mutable.HashMap[(LBJLearnerEquivalent, Any), CategoricalVariable[String]],
    factors: mutable.ListBuffer[Factor],
    variables: mutable.HashSet[CategoricalVariable[String]]
  ): Unit = {
    // No-op
  }
}