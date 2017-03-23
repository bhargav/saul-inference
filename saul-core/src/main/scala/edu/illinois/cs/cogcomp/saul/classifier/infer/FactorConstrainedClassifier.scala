/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer

import edu.illinois.cs.cogcomp.lbjava.classify.TestDiscrete
import edu.illinois.cs.cogcomp.saul.classifier.Learnable
import edu.illinois.cs.cogcomp.saul.datamodel.edge.Edge
import edu.illinois.cs.cogcomp.saul.datamodel.node.Node
import edu.illinois.cs.cogcomp.saul.util.Logging

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
      val prediction = apply(instance)

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

        tester.reportPrediction(prediction, gold)
      })
    }

    headInstances.foreach(testInternal(_, tester))

    tester.printPerformance(System.out)
  }
}
