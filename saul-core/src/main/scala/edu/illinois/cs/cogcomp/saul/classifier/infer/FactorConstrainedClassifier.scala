/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer

import cc.factorie.infer.{ BP, BPMaxProductRing, BPSummary }
import cc.factorie.{ DenseTensor1, Factor }
import cc.factorie.model.{ DotTemplateWithStatistics1, ItemizedModel, Parameters }
import cc.factorie.variable._
import edu.illinois.cs.cogcomp.lbjava.classify.{ ScoreSet, TestDiscrete }
import edu.illinois.cs.cogcomp.lbjava.learn.Softmax
import edu.illinois.cs.cogcomp.saul.classifier.Learnable
import edu.illinois.cs.cogcomp.saul.datamodel.edge.Edge
import edu.illinois.cs.cogcomp.saul.datamodel.node.Node
import edu.illinois.cs.cogcomp.saul.lbjrelated.LBJLearnerEquivalent
import edu.illinois.cs.cogcomp.saul.util.Logging

import scala.collection.JavaConversions._
import scala.collection.mutable

abstract class FactorConstrainedClassifier[T <: AnyRef, HEAD <: AnyRef](val baseNode: Node[T]) extends Logging {

  /** Base classifier to use */
  def onClassifier: Learnable[T]

  protected def subjectTo: Option[Constraint[HEAD]] = None

  protected def pathToHead: Option[Edge[T, HEAD]] = None

  def apply(instance: T): String = onClassifier(instance)

  private def build(head: HEAD, instance: T): String = {
    onClassifier(instance)
  }

  private def findHead(instance: T): Option[HEAD] = {
    if (pathToHead.isEmpty) {
      Some(instance.asInstanceOf[HEAD])
    } else {
      val headCandidates = pathToHead.get.forward.neighborsOf(instance).toSet

      if (headCandidates.isEmpty) {
        logger.trace("Failed to find head")
      } else if (headCandidates.size > 1) {
        logger.trace("Found too many heads; Investigate!!")
      }

      headCandidates.headOption
    }
  }

  private def getCandidates(head: HEAD): Seq[T] = {
    pathToHead.map(edge => edge.backward.neighborsOf(head).toSeq)
      .getOrElse(Seq(head.asInstanceOf[T]))
  }

  private def testInternal(head: HEAD, performanceReporter: TestDiscrete): Unit = {
    getCandidates(head).foreach({ instance: T =>
      val gold = onClassifier.getLabeler.discreteValue(instance)
      val basePrediction = apply(instance)
      val softmax = new Softmax()

      val labelSet: ScoreSet = softmax.normalize(onClassifier.classifier.scores(instance))
      val labels: List[String] = labelSet.values().map(_.asInstanceOf[String]).toList

      // Create a dummy domain for the variable
      object DummyDomain extends CategoricalDomain[String](labels)
      class DummyVariable(label: String, val name: String) extends CategoricalVariable[String] {
        override def domain: CategoricalDomain[String] = DummyDomain
      }

      val family = new DotTemplateWithStatistics1[DummyVariable] with Parameters {
        val weights = Weights(new DenseTensor1(DummyDomain.size))
      }

      DummyDomain.foreach({ domainItem: CategoricalValue[String] =>
        val score = math.log(labelSet.getScore(domainItem.category).score)
        family.weights.value(domainItem.intValue) = score
      })

      val factors = new mutable.ListBuffer[Factor]()
      val variable = new DummyVariable(basePrediction, "Testing")
      factors ++= family.factors(variable)

      val model = new ItemizedModel(factors)
      val fg = BPSummary(Set(variable), BPMaxProductRing, model)
      BP.inferLoopyMax(fg)

      val maxIndex = fg.marginal(variable).proportions.maxIndex
      val prediction = DummyDomain(maxIndex).category

      performanceReporter.reportPrediction(prediction, gold)
    })
  }

  def test(): Unit = {
    val testingSet = onClassifier.node.getTestingInstances.toSet
    val headInstances = testingSet.flatMap(findHead).toSeq.distinct

    val tester = new TestDiscrete()

    val unMappedInstances = testingSet.diff(headInstances.flatMap(getCandidates).toSet)
    if (unMappedInstances.nonEmpty) {
      logger.info("Unmapped instances")

      unMappedInstances.foreach({ instance: T =>
        val gold = onClassifier.getLabeler.discreteValue(instance)
        val prediction = apply(instance)

        logger.info("Unmapped report!")
        tester.reportPrediction(prediction, gold)
      })
    }

    headInstances.foreach(testInternal(_, tester))

    tester.printPerformance(System.out)
  }
}
