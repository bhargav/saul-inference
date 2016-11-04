/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer

import edu.illinois.cs.cogcomp.infer.ilp.{ GurobiHook, ILPSolver, OJalgoHook }
import edu.illinois.cs.cogcomp.lbjava.classify.TestDiscrete
import edu.illinois.cs.cogcomp.lbjava.infer.BalasHook
import edu.illinois.cs.cogcomp.saul.classifier._
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

  def onClassifier: LBJLearnerEquivalent
  protected def subjectTo: Option[Constraint[HEAD]] = None

  protected sealed trait SolverType
  protected case object Gurobi extends SolverType
  protected case object OJAlgo extends SolverType
  protected case object Balas extends SolverType
  protected def solverType: SolverType = OJAlgo

  protected sealed trait OptimizationType
  protected case object Max extends OptimizationType
  protected case object Min extends OptimizationType
  protected def optimizationType: OptimizationType = Max

  private val inferenceManager = new InferenceManager()

  def getClassSimpleNameForClassifier = this.getClass.getSimpleName

  def apply(t: T): String = build(t)

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

  def findHead(x: T): Option[HEAD] = {
    if (tType.equals(headType) || pathToHead.isEmpty) {
      Some(x.asInstanceOf[HEAD])
    } else {
      val l = pathToHead.get.forward.neighborsOf(x).toSet.toSeq
      l.length match {
        case 0 =>
          logger.error("Failed to find head")
          None
        case 1 =>
          logger.trace(s"Found head ${l.head} for child $x")
          Some(l.head)
        case _ =>
          logger.error("Found too many heads; this is usually because some instances belong to multiple 'head's")
          Some(l.head)
      }
    }
  }

  private def getSolverInstance: ILPSolver = solverType match {
    case OJAlgo => new OJalgoHook()
    case Gurobi => new GurobiHook()
    case Balas => new BalasHook()
    case _ => throw new Exception("Hook not found! ")
  }

  def build(t: T): String = {
    findHead(t) match {
      case Some(head) => build(head, t)
      case None => onClassifier.classifier.discreteValue(t)
    }
  }

  def cacheKey[U](u: U): String = u.toString

  def getInstancesInvolvedInProblem(constraintsOpt: Option[Constraint[_]]): Option[Set[_]] = {
    constraintsOpt.map { constraint => getInstancesInvolved(constraint) }
  }

  /** given a head instance, produces a constraint based off of it */
  def instantiateConstraintGivenInstance(head: HEAD): Option[Constraint[_]] = {
    // look at only the first level; if it is PerInstanceConstraint, replace it.
    subjectTo.map {
      case constraint: PerInstanceConstraint[HEAD] => constraint.sensor(head)
      case constraint: Constraint[_] => constraint
    }
  }

  def getInstancesInvolved(constraint: Constraint[_]): Set[_] = {
    constraint match {
      case c: PropositionalEqualityConstraint[_] =>
        Set(c.instanceOpt.get)
      case c: PairConjunction[_, _] =>
        getInstancesInvolved(c.c1) ++ getInstancesInvolved(c.c2)
      case c: PairDisjunction[_, _] =>
        getInstancesInvolved(c.c1) ++ getInstancesInvolved(c.c2)
      case c: Negation[_] =>
        getInstancesInvolved(c.p)
      case c: AtLeast[_, _] =>
        c.constraints.foldRight(Set[Any]()) {
          case (singleConstraint, ins) =>
            ins union getInstancesInvolved(singleConstraint).asInstanceOf[Set[Any]]
        }
      case c: AtMost[_, _] =>
        c.constraints.foldRight(Set[Any]()) {
          case (singleConstraint, ins) =>
            ins union getInstancesInvolved(singleConstraint).asInstanceOf[Set[Any]]
        }
      case c: ForAll[_, _] =>
        c.constraints.foldRight(Set[Any]()) {
          case (singleConstraint, ins) =>
            ins union getInstancesInvolved(singleConstraint).asInstanceOf[Set[Any]]
        }
      case c: Exactly[_, _] =>
        c.constraints.foldRight(Set[Any]()) {
          case (singleConstraint, ins) =>
            ins union getInstancesInvolved(singleConstraint).asInstanceOf[Set[Any]]
        }
      case c: EstimatorPairEqualityConstraint[_] =>
        Set(c.instance)
      case c: InstancePairEqualityConstraint[_] =>
        Set(c.instance1, c.instance2Opt.get)
      case c: Implication[_, _] =>
        throw new Exception("this constraint should have been rewritten in terms of other constraints. ")
    }
  }

  private def build(head: HEAD, t: T)(implicit d: DummyImplicit): String = {
    val constraintsOpt = instantiateConstraintGivenInstance(head)
    val instancesInvolved = getInstancesInvolvedInProblem(constraintsOpt)
    if (constraintsOpt.isDefined && instancesInvolved.get.isEmpty) {
      logger.warn("there are no instances associated with the constraints. It might be because you have defined " +
        "the constraints with 'val' modifier, instead of 'def'.")
    }
    val instanceIsInvolvedInConstraint = instancesInvolved.exists { set =>
      set.exists {
        case x: T => x == t
        case everythingElse => false
      }
    }
    if (instanceIsInvolvedInConstraint) {
      val mainCacheKey = instancesInvolved.map(cacheKey(_)).toSeq.sorted.mkString("*") + onClassifier.toString + constraintsOpt
      val resultOpt = inferenceManager.cachedResults.get(mainCacheKey)
      resultOpt match {
        case Some((cachedSolver, cachedClassifier, cachedEstimatorToSolverLabelMap)) =>
          getInstanceLabel(t, cachedSolver, cachedClassifier, cachedEstimatorToSolverLabelMap)
        case None =>
          // create a new solver instance
          val solver = getSolverInstance
          solver.setMaximize(optimizationType == Max)

          // populate the instances connected to head
          val candidates = getCandidates(head)
          inferenceManager.addVariablesToInferenceProblem(candidates, onClassifier, solver)

          constraintsOpt.foreach { constraints =>
            val inequalities = inferenceManager.processConstraints(constraints, solver)
            inequalities.foreach { ineq =>
              solver.addGreaterThanConstraint(ineq.x, ineq.a, ineq.b)
            }
          }

          solver.solve()
          if (!solver.isSolved) {
            logger.warn("Instance not solved . . . ")
          }

          inferenceManager.cachedResults.put(mainCacheKey, (solver, onClassifier, inferenceManager.estimatorToSolverLabelMap))

          getInstanceLabel(t, solver, onClassifier, inferenceManager.estimatorToSolverLabelMap)
      }
    } else {
      // if the instance doesn't involve in any constraints, it means that it's a simple non-constrained problem.
      logger.info("getting the label with the highest score . . . ")
      onClassifier.classifier.scores(t).highScoreValue()
    }
  }

  def getInstanceLabel(t: T, solver: ILPSolver,
    classifier: LBJLearnerEquivalent,
    estimatorToSolverLabelMap: mutable.Map[LBJLearnerEquivalent, mutable.Map[_, Seq[(Int, String)]]]): String = {
    val estimatorSpecificMap = estimatorToSolverLabelMap(classifier).asInstanceOf[mutable.Map[T, Seq[(Int, String)]]]
    estimatorSpecificMap.get(t) match {
      case Some(indexLabelPairs) =>
        val values = indexLabelPairs.map {
          case (ind, _) => solver.getIntegerValue(ind)
        }
        // exactly one label should be active; if not, [probably] the inference has been infeasible and
        // it is not usable, in which case we make direct calls to the non-constrained classifier.
        if (values.sum == 1) {
          indexLabelPairs.collectFirst {
            case (ind, label) if solver.getIntegerValue(ind) == 1.0 => label
          }.get
        } else {
          onClassifier.classifier.scores(t).highScoreValue()
        }
      case None => throw new Exception("instance is not cached ... weird! :-/ ")
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
    * @return Seq of ???
    */
  def test(testData: Iterable[T] = null, outFile: String = null, outputGranularity: Int = 0, exclude: String = ""): Results = {
    val testReader = if (testData == null) deriveTestInstances else testData
    val tester = new TestDiscrete()
    testReader.foreach { instance =>
      val label = onClassifier.getLabeler.discreteValue(instance)
      val prediction = build(instance)
      tester.reportPrediction(prediction, label)
    }
    val perLabelResults = tester.getLabels.map {
      label =>
        ResultPerLabel(label, tester.getF1(label), tester.getPrecision(label), tester.getRecall(label),
          tester.getAllClasses, tester.getLabeled(label), tester.getPredicted(label), tester.getCorrect(label))
    }
    val overallResultArray = tester.getOverallStats()
    val overallResult = OverallResult(overallResultArray(0), overallResultArray(1), overallResultArray(2))
    println("overallResult =" + overallResult)
    Results(perLabelResults, ClassifierUtils.getAverageResults(perLabelResults), overallResult)
  }
}