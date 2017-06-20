/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer

import java.util.Date

import edu.illinois.cs.cogcomp.core.io.LineIO
import edu.illinois.cs.cogcomp.lbjava.classify.{ ScoreSet, TestDiscrete }
import edu.illinois.cs.cogcomp.saul.classifier._
import edu.illinois.cs.cogcomp.saul.classifier.infer.solver._
import edu.illinois.cs.cogcomp.saul.datamodel.edge.Edge
import edu.illinois.cs.cogcomp.saul.lbjrelated.LBJLearnerEquivalent
import edu.illinois.cs.cogcomp.saul.util.Logging

import scala.collection.{ Iterable, Seq, mutable }
import scala.reflect.ClassTag

abstract class ConstrainedClassifier[T <: AnyRef, HEAD <: AnyRef](
  implicit
  val tType: ClassTag[T],
  implicit val headType: ClassTag[HEAD]
) extends Logging {

  type InstanceType = T
  type HeadType = HEAD

  def onClassifier: LBJLearnerEquivalent
  protected def subjectTo: Option[Constraint[HEAD]] = None
  protected def solverType: SolverType = OJAlgo
  protected def useCaching: Boolean = true

  protected def optimizationType: OptimizationType = Max

  // This should be lazy so that correct solverType is passed in.
  protected lazy val inferenceSolver: InferenceSolver[T, HEAD] = new ILPInferenceSolver[T, HEAD](solverType, optimizationType, onClassifier, useCaching)

  def getClassSimpleNameForClassifier: String = this.getClass.getSimpleName

  /** The function is used to filter the generated candidates from the head object; remember that the inference starts
    * from the head object. This function finds the objects of type [[T]] which are connected to the target object of
    * type [[HEAD]]. If we don't define [[filter]], by default it returns all objects connected to [[HEAD]].
    * The filter is useful for the `JointTraining` when we go over all global objects and generate all contained object
    * that serve as examples for the basic classifiers involved in the `JoinTraining`. It is possible that we do not
    * want to use all possible candidates but some of them, for example when we have a way to filter the negative
    * candidates, this can come in the filter.
    */
  protected def filter(t: T, head: HEAD): Boolean = true

  /** The [[pathToHead]] returns only one object of type HEAD, if there are many of them i.e. `Iterable[HEAD]` then it
    * simply returns the head of the [[Iterable]]
    */
  protected def pathToHead: Option[Edge[T, HEAD]] = None

  private def deriveTestInstances: Iterable[T] = {
    pathToHead.map(edge => edge.from)
      .orElse({
        onClassifier match {
          case clf: Learnable[T] => Some(clf.node)
          case _ => logger.error("pathToHead is not provided and the onClassifier is not a Learnable!"); None
        }
      })
      .map(node => node.getTestingInstances)
      .getOrElse(Iterable.empty)
  }

  def getCandidates(head: HEAD): Seq[T] = {
    if (tType.equals(headType) || pathToHead.isEmpty) {
      Seq(head.asInstanceOf[T])
    } else {
      val l = pathToHead.get.backward.neighborsOf(head)
      l.size match {
        case 0 =>
          logger.error("Failed to find part")
          Seq.empty[T]
        case _ => l.filter(filter(_, head)).toSeq
      }
    }
  }

  private def findHead(x: T): Option[HEAD] = {
    if (tType.equals(headType) || pathToHead.isEmpty) {
      Some(x.asInstanceOf[HEAD])
    } else {
      val l = pathToHead.get.forward.neighborsOf(x).toSet.toSeq
      l.length match {
        case 0 =>
          logger.trace("Failed to find head")
          None
        case 1 =>
          logger.trace(s"Found head ${l.head} for child $x")
          Some(l.head)
        case _ =>
          logger.trace("Found too many heads; this is usually because some instances belong to multiple 'head's")
          Some(l.head)
      }
    }
  }

  /** given an instance */
  def apply(instance: InstanceType): String = {
    val headInstance = findHead(instance)

    if (headInstance.isEmpty) {
      onClassifier.classifier.discreteValue(instance)
    } else {
      val finalAssignments = build(headInstance.get, instance)

      val finalInstanceAssignment = finalAssignments.find(_.learner == onClassifier)
        .flatMap(_.get(instance))
        .map(_.highScoreValue())
        .getOrElse({
          logger.trace("Instance or Classifier not found by the inference solver!")
          onClassifier.classifier.scores(instance).highScoreValue()
        })

      finalInstanceAssignment
    }
  }

  def apply(instances: Iterable[InstanceType], progressPeriod: Int = 0): Map[InstanceType, String] = {
    val instanceLabelMap = new mutable.HashMap[InstanceType, String]()

    instances.zipWithIndex.foreach({ case (instance, idx) =>
      if (progressPeriod > 0 && idx % progressPeriod == 0) {
        logger.info(s"Processed $idx instances.")
      }

      if (!instanceLabelMap.contains(instance)) {
        val headInstance = findHead(instance)

        if (headInstance.isEmpty) {
          instanceLabelMap.put(instance, onClassifier.classifier.discreteValue(instance))
        } else {
          val assignments = build(headInstance.get, instance)
          val finalAssignment = assignments.find(_.learner == onClassifier).get
          finalAssignment.foreach({
            case (scoredInstance: Any, scoreset: ScoreSet) =>
              instanceLabelMap.put(scoredInstance.asInstanceOf[InstanceType], scoreset.highScoreValue())
          })
        }
      }
    })

    instanceLabelMap.toMap
  }

  private def getClassifierAndInstancesInvolvedInProblem(constraintsOpt: Option[Constraint[_]]): Option[Set[(LBJLearnerEquivalent, _)]] = {
    constraintsOpt.map { constraint => getClassifierAndInstancesInvolved(constraint) }
  }

  /** given a head instance, produces a constraint based off of it */
  private def instantiateConstraintGivenInstance(head: HEAD): Option[Constraint[_]] = {
    // look at only the first level; if it is PerInstanceConstraint, replace it.
    subjectTo.map {
      case constraint: PerInstanceConstraint[HEAD] => constraint.sensor(head)
      case constraint: Constraint[_] => constraint
    }
  }

  private def getClassifierAndInstancesInvolved(constraint: Constraint[_]): Set[(LBJLearnerEquivalent, _)] = {
    constraint match {
      case c: PropositionalEqualityConstraint[_] =>
        Set((c.estimator, c.instanceOpt.get))
      case c: PairConjunction[_, _] =>
        getClassifierAndInstancesInvolved(c.c1) ++ getClassifierAndInstancesInvolved(c.c2)
      case c: PairDisjunction[_, _] =>
        getClassifierAndInstancesInvolved(c.c1) ++ getClassifierAndInstancesInvolved(c.c2)
      case c: Implication[_, _] =>
        getClassifierAndInstancesInvolved(c.p) ++ getClassifierAndInstancesInvolved(c.q)
      case c: Negation[_] =>
        getClassifierAndInstancesInvolved(c.p)
      case c: ConstraintCollections[_, _] =>
        c.constraints.foldRight(Set[(LBJLearnerEquivalent, Any)]()) {
          case (singleConstraint, ins) =>
            ins union getClassifierAndInstancesInvolved(singleConstraint)
        }
      case c: EstimatorPairEqualityConstraint[_] =>
        if (c.estimator2Opt.nonEmpty) {
          Set((c.estimator1, c.instance), (c.estimator2Opt.get, c.instance))
        } else {
          Set((c.estimator1, c.instance))
        }
      case c: InstancePairEqualityConstraint[_] =>
        if (c.instance2Opt.nonEmpty) {
          Set((c.estimator, c.instance1), (c.estimator, c.instance2Opt.get))
        } else {
          Set((c.estimator, c.instance1))
        }
      case _ =>
        throw new Exception("Unknown constraint exception! This constraint should have been rewritten in terms of other constraints. ")
    }
  }

  /** Builds the Constraint Satisfaction Problem and returns the assignment for the instance node.
    *
    * @param head
    * @param instance
    * @return
    */
  private def build(head: HEAD, instance: T): Seq[Assignment] = {
    val constraintsOpt = instantiateConstraintGivenInstance(head)
    val classifiersAndInstancesInvolved = getClassifierAndInstancesInvolvedInProblem(constraintsOpt)

    if (constraintsOpt.isDefined && classifiersAndInstancesInvolved.get.isEmpty) {
      logger.warn("there are no instances associated with the constraints. It might be because you have defined " +
        "the constraints with 'val' modifier, instead of 'def'.")
    }

    val classifierAndInstanceInvolved = classifiersAndInstancesInvolved.exists({ set =>
      set.exists({
        case (c: LBJLearnerEquivalent, x: T) => (onClassifier == c) && (x == instance)
        case _ => false
      })
    })

    if (classifierAndInstanceInvolved) {
      val localAssignments = classifiersAndInstancesInvolved.map({ set =>
        set.groupBy(_._1).map({
          case (classifier: LBJLearnerEquivalent, classifierInstanceSet: Set[(LBJLearnerEquivalent, Any)]) =>
            val assignment = Assignment(classifier)
            classifierInstanceSet.map(_._2).foreach({ instance: Any =>
              assignment += ((instance, classifier.classifier.scores(instance)))
            })
            assignment
        })
      })

      inferenceSolver.solve(
        constraintsOpt = constraintsOpt,
        priorAssignment = localAssignments.get.toSeq
      )
    } else {
      // if the instance doesn't involve in any constraints, it means that it's a simple non-constrained problem.
      logger.trace("getting the label with the highest score . . . ")

      val assignment = Assignment(onClassifier)
      assignment.put(instance, onClassifier.classifier.scores(instance))
      Seq(assignment)
    }
  }

  /** Test Constrained Classifier with automatically derived test instances.
    *
    * @return A [[Results]] object
    */
  def test(): Results = {
    test(deriveTestInstances)
  }

  /** Test with given data, use internally
    *
    * @param testData if the collection of data (which is and Iterable of type T) is not given it is derived from the
    * data model based on its type
    * @param exclude it is the label that we want to exclude for evaluation, this is useful for evaluating the multi-class
    * classifiers when we need to measure overall F1 instead of accuracy and we need to exclude the negative class
    * @param outFile The file to write the predictions (can be `null`)
    * @return A [[Results]] object
    */
  def test(testData: Iterable[T] = null, outFile: String = null, outputGranularity: Int = 0, exclude: String = ""): Results = {
    val testReader = if (testData == null) deriveTestInstances else testData
    val tester = new TestDiscrete()
    if (exclude.nonEmpty) tester.addNull(exclude)

    val predictions = apply(testReader, outputGranularity)

    predictions.foreach {
      case (instance, prediction) =>
        val gold = onClassifier.getLabeler.discreteValue(instance)
        tester.reportPrediction(prediction, gold)

        // Append the predictions to a file (if the outFile parameter is given)
        if (outFile != null) {
          try {
            val line = "Example " + instance + "\tprediction:\t" + prediction + "\t gold:\t" + gold + "\t" + (if (gold.equals(prediction)) "correct" else "incorrect")
            LineIO.append(outFile, line);
          } catch {
            case e: Exception => e.printStackTrace()
          }
        }
    }

    println() // for an extra empty line, for visual convenience :)
    tester.printPerformance(System.out)

    val perLabelResults = tester.getLabels.map {
      label =>
        ResultPerLabel(label, tester.getF1(label), tester.getPrecision(label), tester.getRecall(label),
          tester.getAllClasses, tester.getLabeled(label), tester.getPredicted(label), tester.getCorrect(label))
    }
    val overallResultArray = tester.getOverallStats()
    val overallResult = OverallResult(overallResultArray(0), overallResultArray(1), overallResultArray(2))
    Results(perLabelResults, ClassifierUtils.getAverageResults(perLabelResults), overallResult)
  }
}
