/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer

import java.util.Date

import edu.illinois.cs.cogcomp.core.io.LineIO
import edu.illinois.cs.cogcomp.lbjava.classify.TestDiscrete
import edu.illinois.cs.cogcomp.saul.classifier._
import edu.illinois.cs.cogcomp.saul.classifier.infer.solver._
import edu.illinois.cs.cogcomp.saul.datamodel.edge.Edge
import edu.illinois.cs.cogcomp.saul.lbjrelated.LBJLearnerEquivalent
import edu.illinois.cs.cogcomp.saul.util.Logging

import scala.collection.{ Iterable, Seq }
import scala.reflect.ClassTag

/** possible solvers to use */
sealed trait SolverType
case object Gurobi extends SolverType
case object OJAlgo extends SolverType
case object Balas extends SolverType

abstract class ConstrainedClassifier[T <: AnyRef, HEAD <: AnyRef](
  implicit
  val tType: ClassTag[T],
  implicit val headType: ClassTag[HEAD]
) extends Logging {

  def onClassifier: LBJLearnerEquivalent
  protected def subjectTo: Option[Constraint[HEAD]] = None
  protected def solverType: SolverType = OJAlgo
  protected def useCaching: Boolean = true

  protected def optimizationType: OptimizationType = Max

  private val inferenceSolver = new ILPInferenceSolver[T, HEAD](solverType, optimizationType, onClassifier)

  def getClassSimpleNameForClassifier = this.getClass.getSimpleName

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

  def apply(t: T): String = {
    findHead(t) match {
      case Some(head) => build(head, t)
      case None => onClassifier.classifier.discreteValue(t)
    }
  }

  private def getInstancesInvolvedInProblem(constraintsOpt: Option[Constraint[_]]): Option[Set[_]] = {
    constraintsOpt.map { constraint => getInstancesInvolved(constraint) }
  }

  private def getClassifiersInvolvedInProblem(constraintsOpt: Option[Constraint[_]]): Option[Set[LBJLearnerEquivalent]] = {
    constraintsOpt.map { constraint => getClassifiersInvolved(constraint) }
  }

  /** given a head instance, produces a constraint based off of it */
  private def instantiateConstraintGivenInstance(head: HEAD): Option[Constraint[_]] = {
    // look at only the first level; if it is PerInstanceConstraint, replace it.
    subjectTo.map {
      case constraint: PerInstanceConstraint[HEAD] => constraint.sensor(head)
      case constraint: Constraint[_] => constraint
    }
  }

  /** find all the instances used in the definiton of the constraint. This is used in caching the results of inference  */
  private def getInstancesInvolved(constraint: Constraint[_]): Set[_] = {
    constraint match {
      case c: PropositionalEqualityConstraint[_] =>
        Set(c.instanceOpt.get)
      case c: PairConjunction[_, _] =>
        getInstancesInvolved(c.c1) ++ getInstancesInvolved(c.c2)
      case c: PairDisjunction[_, _] =>
        getInstancesInvolved(c.c1) ++ getInstancesInvolved(c.c2)
      case c: Negation[_] =>
        getInstancesInvolved(c.p)
      case c: ConstraintCollections[_, _] =>
        c.constraints.foldRight(Set[Any]()) {
          case (singleConstraint, ins) =>
            ins union getInstancesInvolved(singleConstraint).asInstanceOf[Set[Any]]
        }
      case c: EstimatorPairEqualityConstraint[_] =>
        Set(c.instance)
      case c: InstancePairEqualityConstraint[_] =>
        Set(c.instance1, c.instance2Opt.get)
      case _ =>
        throw new Exception("Unknown constraint exception! This constraint should have been rewritten in terms of other constraints. ")
    }
  }

  /** find all the classifiers involved in the definition of the constraint. This is used for caching of inference */
  private def getClassifiersInvolved(constraint: Constraint[_]): Set[LBJLearnerEquivalent] = {
    constraint match {
      case c: PropositionalEqualityConstraint[_] =>
        Set(c.estimator)
      case c: PairConjunction[_, _] =>
        getClassifiersInvolved(c.c1) ++ getClassifiersInvolved(c.c2)
      case c: PairDisjunction[_, _] =>
        getClassifiersInvolved(c.c1) ++ getClassifiersInvolved(c.c2)
      case c: Negation[_] =>
        getClassifiersInvolved(c.p)
      case c: ConstraintCollections[_, _] =>
        c.constraints.foldRight(Set[LBJLearnerEquivalent]()) {
          case (singleConstraint, ins) =>
            ins union getClassifiersInvolved(singleConstraint)
        }
      case c: EstimatorPairEqualityConstraint[_] =>
        Set(c.estimator1, c.estimator2Opt.get)
      case c: InstancePairEqualityConstraint[_] =>
        Set(c.estimator)
      case _ =>
        throw new Exception("Unknown constraint exception! This constraint should have been rewritten in terms of other constraints. ")
    }
  }

  private def build(head: HEAD, t: T)(implicit d: DummyImplicit): String = {
    val constraintsOpt = instantiateConstraintGivenInstance(head)
    val instancesInvolved = getInstancesInvolvedInProblem(constraintsOpt)
    val classifiersInvolved = getClassifiersInvolvedInProblem(constraintsOpt)
    if (constraintsOpt.isDefined && instancesInvolved.get.isEmpty) {
      logger.warn("there are no instances associated with the constraints. It might be because you have defined " +
        "the constraints with 'val' modifier, instead of 'def'.")
    }
    val instanceIsInvolvedInConstraint = instancesInvolved.exists { set =>
      set.exists {
        case x: T => x == t
        case _ => false
      }
    }
    val classifierIsInvolvedInProblem = classifiersInvolved.exists { classifierSet =>
      classifierSet.exists {
        case c: LBJLearnerEquivalent => onClassifier == c
        case _ => false
      }
    }
    if (instanceIsInvolvedInConstraint & classifierIsInvolvedInProblem) {
      /** The following cache-key is very important, as it defines what to and when to cache the results of the inference.
        * The first term encodes the instances involved in the constraint, after propositionalization, and the second term
        * contains pure definition of the constraint before any propositionalization.
        */
      val cacheKey = instancesInvolved.map(_.toString).toSeq.sorted.mkString("*") + constraintsOpt
      val candidates = getCandidates(head)

      inferenceSolver.solve(cacheKey, instance = t, constraintsOpt, candidates)
    } else {
      // if the instance doesn't involve in any constraints, it means that it's a simple non-constrained problem.
      logger.trace("getting the label with the highest score . . . ")
      onClassifier.classifier.scores(t).highScoreValue()
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
    testReader.zipWithIndex.foreach {
      case (instance, idx) =>
        val gold = onClassifier.getLabeler.discreteValue(instance)
        val prediction = apply(instance)
        tester.reportPrediction(prediction, gold)

        if (outputGranularity > 0 && idx % outputGranularity == 0) {
          println(idx + " examples tested at " + new Date())
        }

        // Append the predictions to a file (if the outFile parameter is given)
        if (outFile != null) {
          try {
            val line = "Example " + idx + "\tprediction:\t" + prediction + "\t gold:\t" + gold + "\t" + (if (gold.equals(prediction)) "correct" else "incorrect")
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
