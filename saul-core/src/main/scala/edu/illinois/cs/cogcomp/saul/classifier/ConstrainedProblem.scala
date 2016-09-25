/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier

import edu.illinois.cs.cogcomp.infer.ilp.{ OJalgoHook, GurobiHook, ILPSolver }
import edu.illinois.cs.cogcomp.lbjava.infer.BalasHook
import edu.illinois.cs.cogcomp.saul.datamodel.edge.Edge
import edu.illinois.cs.cogcomp.saul.datamodel.node.Node
import edu.illinois.cs.cogcomp.saul.lbjrelated.LBJLearnerEquivalent
import edu.illinois.cs.cogcomp.saul.util.Logging

import scala.collection.{ mutable, Iterable }
import scala.reflect.ClassTag

abstract class ConstrainedProblem[T <: AnyRef, HEAD <: AnyRef](
  implicit
  val tType: ClassTag[T],
  implicit val headType: ClassTag[HEAD]
) extends Logging {
  import ConstrainedProblem._

  protected def estimator: LBJLearnerEquivalent
  protected def constraintsOpt: Option[SaulConstraint[HEAD]] = None

  protected sealed trait SolverType
  protected case object Gurobi extends SolverType
  protected case object OJAlgo extends SolverType
  protected case object Balas extends SolverType
  protected def solverType: SolverType = OJAlgo

  protected sealed trait OptimizationType
  protected case object Max extends OptimizationType
  protected case object Min extends OptimizationType
  protected def optimizationType: OptimizationType = Max

  def apply(t: T): String = ""

  /** The function is used to filter the generated candidates from the head object; remember that the inference starts
    * from the head object. This function finds the objects of type `T` which are connected to the target object of
    * type `HEAD`. If we don't define `filter`, by default it returns all objects connected to `HEAD`.
    * The filter is useful for the `JointTraining` when we go over all global objects and generate all contained object
    * that serve as examples for the basic classifiers involved in the `JoinTraining`. It is possible that we do not
    * want to use all possible candidates but some of them, for example when we have a way to filter the negative
    * candidates, this can come in the filter.
    */
  protected def filter(t: T, head: HEAD): Boolean = true

  /** The `pathToHead` returns only one object of type HEAD, if there are many of them i.e. `Iterable[HEAD]` then it
    * simply returns the `head` of the `Iterable`
    */
  protected def pathToHead: Option[Edge[T, HEAD]] = None

  private def deriveTestInstances: Iterable[T] = pathToHead.map(_.from.getTestingInstances).getOrElse(Iterable.empty)

  private def getCandidates(head: HEAD): Seq[T] = {
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
          logger.error("Warning: Failed to find head")
          None
        case 1 =>
          logger.info(s"Found head ${l.head} for child $x")
          Some(l.head)
        case _ =>
          logger.warn("Find too many heads; this is usually because some instances belong to multiple 'head's")
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

  def build(t: T): Unit = {
    findHead(t) match {
      case Some(head) => build(head)
      case None => // do nothing
    }
  }

  def build(head: HEAD)(implicit d: DummyImplicit): Unit = {
    // create a new solver instance
    val solver = getSolverInstance

    // populate the instances connected to head
    val candidates = getCandidates(head)
    addVariablesToInferenceProblem(candidates, estimator, solver)

    // populate the constraints and relevant variables
    constraintsOpt.foreach {
      case constraints =>
        val inequalities = processConstraints(head, constraints, solver)
        inequalities.set.foreach { inequality =>
          solver.addLessThanConstraint(inequality.x, inequality.a, inequality.b)
        }
    }

    solver.solve()
    println("# of candidates: " + candidates.length)
    println("length of instanceLabelVarMap: " + estimatorToSolverLabelMap.size)
    println("length of instanceLabelVarMap: " + estimatorToSolverLabelMap.get(estimator).get.size)
    candidates.foreach { c =>
      val estimatorSpecificMap = estimatorToSolverLabelMap.get(estimator).get.asInstanceOf[mutable.Map[T, Seq[(Int, String)]]]
      estimatorSpecificMap.get(c) match {
        case Some(a) => a.foreach {
          case (ind, label) =>
            println("Instance: " + c)
            println(s"label:$label -> ")
            println(s"int:$ind -> ")
            println("solver.getIntegerValue: " + solver.getIntegerValue(ind))
        }
        case None => throw new Exception("instance is not cached ... weird! :-/ ")
      }
    }

    /*val name = head.toString + head.hashCode()
    if(InferenceManager.problemNames.contains(name)){

    } else {
      logger.warn(s"Inference $name has not been cached; running inference . . . ")
    }*/
  }

  //def solve(): Boolean = ??? /// solver.solve()

  /** Test Constrained Classifier with automatically derived test instances.
    *
    * @return Seq of ???
    */
  /*  def test(): Results = {
    test(deriveTestInstances)
  }

  /** Test with given data, use internally
    *
    * @param testData if the collection of data (which is and Iterable of type T) is not given it is derived from the data model based on its type
    * @param exclude it is the label that we want to exclude for evaluation, this is useful for evaluating the multi-class classifiers when we need to measure overall F1 instead of accuracy and we need to exclude the negative class
    * @param outFile The file to write the predictions (can be `null`)
    * @return Seq of ???
    */
  def test(testData: Iterable[T] = null, outFile: String = null, outputGranularity: Int = 0, exclude: String = ""): Results = {
    println()

    val testReader = new IterableToLBJavaParser[T](if (testData == null) deriveTestInstances else testData)
    testReader.reset()

    val tester: TestDiscrete = new TestDiscrete()
    TestWithStorage.test(tester, classifier, onClassifier.getLabeler, testReader, outFile, outputGranularity, exclude)
    val perLabelResults = tester.getLabels.map {
      label =>
        ResultPerLabel(label, tester.getF1(label), tester.getPrecision(label), tester.getRecall(label),
          tester.getAllClasses, tester.getLabeled(label), tester.getPredicted(label), tester.getCorrect(label))
    }
    val overalResultArray = tester.getOverallStats()
    val overalResult = OverallResult(overalResultArray(0), overalResultArray(1), overalResultArray(2))
    Results(perLabelResults, ClassifierUtils.getAverageResults(perLabelResults), overalResult)
  }*/
}

object ConstrainedProblem {
  /*
  def processConstraints2[V](instance: V, saulConstraint: SaulConstraint[V], solver: ILPSolver): Unit = {

    saulConstraint match {
      case c: SaulFirstOrderConstraint[V] => // do nothing
      case c: SaulPropositionalConstraint[V] =>
        addVariablesToInferenceProblem(Seq(instance), c.estimator, solver)
    }

    saulConstraint match {
      case c: SaulPropositionalEqualityConstraint[V] =>
        // estimates per instance
        val estimatorScoresMap = estimatorToSolverLabelMap.get(c.estimator).get.asInstanceOf[mutable.Map[V, Seq[(Int, String)]]]
        val (indices, labels) = estimatorScoresMap.get(instance).get.unzip
        assert(
          c.inequalityValOpt.isEmpty && c.equalityValOpt.isEmpty,
          s"the equality constraint $c is not completely defined"
        )
        assert(
          c.inequalityValOpt.isDefined && c.equalityValOpt.isDefined,
          s"the equality constraint $c has values for both equality and inequality"
        )
        if (c.equalityValOpt.isDefined) {
          // first make sure the target value is valid
          require(
            c.estimator.classifier.allowableValues().toSet.contains(c.equalityValOpt.get),
            s"The target value ${c.equalityValOpt} is not a valid value for classifier ${c.estimator}"
          )
          val labelIndexOpt = labels.zipWithIndex.collectFirst { case (label, idx) if label == c.equalityValOpt.get => idx }
          val labelIndex = labelIndexOpt.getOrElse(
            throw new Exception()
          )
          val coeffs = Array.fill(indices.length) { 0.0 }
          coeffs(labelIndex) = 1.0
          solver.addEqualityConstraint(indices.toArray, coeffs, 1)
        } else {
          require(
            c.estimator.classifier.allowableValues().toSet.contains(c.inequalityValOpt.get),
            s"The target value ${c.inequalityValOpt} is not a valid value for classifier ${c.estimator}"
          )
          val labelIndexOpt = labels.zipWithIndex.collectFirst { case (label, idx) if label == c.inequalityValOpt.get => idx }
          val labelIndex = labelIndexOpt.getOrElse(
            throw new Exception()
          )
          val coeffs = Array.fill(1) { 1.0 }
          solver.addEqualityConstraint(Array(indices(labelIndex)), coeffs, 0)
        }
      case c: SaulConjunction[V] =>
      case c: SaulDisjunction[V] =>
      case c: SaulImplication[V, _] =>
      case c: SaulNegation[V] =>
      case c: SaulFirstOrderDisjunctionConstraint2[V, _] =>
      case c: SaulFirstOrderConjunctionConstraint2[V, _] =>
      case c: SaulFirstOrderAtLeastConstraint2[V, _] =>
      case c: SaulFirstOrderAtMostConstraint2[V, _] =>
      // case   c: SaulConstraint[T] =>
      // case c: SaulFirstOrderConstraint[T] =>
      // case c: SaulPropositionalConstraint[T] =>
    }
  }
*/

  // ax >= b
  case class ILPInequality(a: Array[Double], x: Array[Int], b: Double)

  case class ILPInequalitySet(set: Set[ILPInequality])

  def processConstraints[V <: Any](instance: V, saulConstraint: SaulConstraint[V], solver: ILPSolver)(implicit tag: ClassTag[V]): ILPInequalitySet = {

    saulConstraint match {
      case c: SaulPropositionalConstraint[V] =>
        addVariablesToInferenceProblem(Seq(instance), c.estimator, solver)
      case _ => // do nothing
    }

    saulConstraint match {
      case c: SaulPropositionalEqualityConstraint[V] =>
        // estimates per instance
        val estimatorScoresMap = estimatorToSolverLabelMap.get(c.estimator).get.asInstanceOf[mutable.Map[V, Seq[(Int, String)]]]
        val (ilpIndices, labels) = estimatorScoresMap.get(instance).get.unzip
        assert(
          c.inequalityValOpt.isEmpty && c.equalityValOpt.isEmpty,
          s"the equality constraint $c is not completely defined"
        )
        assert(
          c.inequalityValOpt.isDefined && c.equalityValOpt.isDefined,
          s"the equality constraint $c has values for both equality and inequality"
        )
        if (c.equalityValOpt.isDefined) {
          // first make sure the target value is valid
          require(
            c.estimator.classifier.allowableValues().toSet.contains(c.equalityValOpt.get),
            s"The target value ${c.equalityValOpt} is not a valid value for classifier ${c.estimator}"
          )
          val labelIndexOpt = labels.zipWithIndex.collectFirst { case (label, idx) if label == c.equalityValOpt.get => idx }
          val x = labelIndexOpt.getOrElse(
            throw new Exception(s"the corresponding index to label ${c.equalityValOpt.get} not found")
          )

          // 1.0 x >= 1 : possible only when x = 1
          val a = Array(1.0)
          val b = 1.0
          ILPInequalitySet(Set(ILPInequality(a, Array(x), b)))
        } else {
          require(
            c.estimator.classifier.allowableValues().toSet.contains(c.inequalityValOpt.get),
            s"The target value ${c.inequalityValOpt} is not a valid value for classifier ${c.estimator}"
          )
          val labelIndexOpt = labels.zipWithIndex.collectFirst { case (label, idx) if label == c.inequalityValOpt.get => idx }
          val x = labelIndexOpt.getOrElse(
            throw new Exception()
          )
          val a = Array(0.1)
          val b = 1.0
          // 0.1 x >= 1 : possible only when x = 0
          ILPInequalitySet(Set(ILPInequality(a, Array(x), b)))
        }
      case c: SaulPairConjunction[V, Any] =>
        val InequalitySystem1 = processConstraints(instance, c.c1, solver)
        val InequalitySystem2 = processConstraints(instance, c.c2, solver)

        // conjunction is simple; you just include all the inequalities
        ILPInequalitySet(InequalitySystem1.set union InequalitySystem2.set)
      case c: SaulPairDisjunction[V, Any] =>
        val InequalitySystem1 = processConstraints(instance, c.c1, solver)
        val InequalitySystem2 = processConstraints(instance, c.c2, solver)

      case c: SaulImplication[V, Any] =>
        val pIneq = processConstraints(instance, c.p, solver)
        val qIneq = processConstraints(instance, c.q, solver)

        // (1) define y in {0, 1}. y = 0, iff pIneq is satisfied
        // --> 1.a: pIneq: ax <= b is satisfied ==> y = 0: y + ax <= b
        // --> 1.b:   ax <= by
        // (2) qIneq should be satisfied, only if y = 1

        val y = solver.addBooleanVariable(1)

      case c: SaulNegation[V] =>
        // change the signs of the coefficients
        val InequalitySystemToBeNegated = processConstraints(instance, c.p, solver)
        val inequalitySet = InequalitySystemToBeNegated.set.map { in =>
          val minusA = in.a.map(-_)
          val minusB = -in.b
          ILPInequality(minusA, in.x, minusB)
        }
        ILPInequalitySet(inequalitySet)
      case c: SaulAtLeast[V, Any] =>
        val InequalitySystemsAtLeast = c.constraints.map { processConstraints(instance, _, solver) }
      case c: SaulAtMost[V, Any] =>
        val InequalitySystemsAtMost = c.constraints.map { processConstraints(instance, _, solver) }
      case c: SaulExists[V, Any] =>
        val InequalitySystemsAtMost = c.constraints.map { processConstraints(instance, _, solver) }
      case c: SaulForAll[V, Any] =>
        val InequalitySystemsAtMost = c.constraints.map { processConstraints(instance, _, solver) }
    }

    // just to make it compile
    ILPInequalitySet(Set.empty)
  }

  // if the estimator has never been seen before, add its labels to the map
  def createEstimatorSpecificCache[V](estimator: LBJLearnerEquivalent): Unit = {
    if (!estimatorToSolverLabelMap.keySet.contains(estimator)) {
      estimatorToSolverLabelMap += (estimator -> mutable.Map[V, Seq[(Int, String)]]())
    }
  }

  def addVariablesToInferenceProblem[V](instances: Seq[V], estimator: LBJLearnerEquivalent, solver: ILPSolver): Unit = {
    createEstimatorSpecificCache(estimator)

    // estimates per instance
    val estimatorScoresMap = estimatorToSolverLabelMap.get(estimator).get.asInstanceOf[mutable.Map[V, Seq[(Int, String)]]]

    // adding the estimates to the solver and to the map
    instances.foreach { c =>
      val confidenceScores = estimator.classifier.scores(c).toArray.map(_.score)
      require(confidenceScores.forall(_ >= 0.0), s"Some of the scores returned by $estimator are below zero.")
      val labels = estimator.classifier.scores(c).toArray.map(_.value)
      val instanceIndexPerLabel = solver.addDiscreteVariable(confidenceScores)
      if (!estimatorScoresMap.contains(c)) {
        estimatorScoresMap += (c -> instanceIndexPerLabel.zip(labels).toSeq)
      }
    }
  }

  import collection._

  // cached results
  //  val cachedResults = mutable.Map[String, mutable.Map[String, Int]]()

  // for each estimator, maps the label of the estimator, to the integer label of the solver
  val estimatorToSolverLabelMap = mutable.Map[LBJLearnerEquivalent, mutable.Map[_, Seq[(Int, String)]]]()

  // for each estimator, maps the integer label of the solver to the label of the estimator
  //  val solverToEstimatorLabelMap = mutable.Map[String, mutable.Map[Int, String]]()
}

import scala.collection.JavaConverters._

object SaulConstraint {
  implicit class LearnerToFirstOrderConstraint(estimator: LBJLearnerEquivalent) {
    def on2[T](newInstance: T)(implicit tag: ClassTag[T]): SaulPropositionalEqualityConstraint[T] = {
      new SaulPropositionalEqualityConstraint[T](estimator, Some(newInstance), None, None)
    }
  }

  implicit def FirstOrderConstraint[T <: AnyRef](coll: Traversable[T]): ConstraintObjWrapper[T] = new ConstraintObjWrapper[T](coll.toSeq)

  implicit def FirstOrderConstraint[T <: AnyRef](coll: Set[T]): ConstraintObjWrapper[T] = new ConstraintObjWrapper[T](coll.toSeq)

  implicit def FirstOrderConstraint[T <: AnyRef](coll: java.util.Collection[T]): ConstraintObjWrapper[T] = new ConstraintObjWrapper[T](coll.asScala.toSeq)

  implicit def FirstOrderConstraint[T <: AnyRef](coll: mutable.LinkedHashSet[T]): ConstraintObjWrapper[T] = new ConstraintObjWrapper[T](coll.toSeq)

  implicit def FirstOrderConstraint[T <: AnyRef](node: Node[T]): ConstraintObjWrapper[T] = new ConstraintObjWrapper[T](node.getAllInstances.toSeq)
}

class ConstraintObjWrapper[T](coll: Seq[T]) {
  def ForAll[U](sensors: T => SaulConstraint[U])(implicit tag: ClassTag[T]): SaulForAll[T, U] = {
    new SaulForAll[T, U](coll.map(sensors))
  }
  def Exists[U](sensors: T => SaulConstraint[U])(implicit tag: ClassTag[T]): SaulExists[T, U] = {
    new SaulExists[T, U](coll.map(sensors))
  }
  def AtLeast[U](k: Int)(sensors: T => SaulConstraint[U])(implicit tag: ClassTag[T]): SaulAtLeast[T, U] = {
    new SaulAtLeast[T, U](coll.map(sensors), k)
  }
  def AtMost[U](k: Int)(sensors: T => SaulConstraint[U])(implicit tag: ClassTag[T]): SaulAtLeast[T, U] = {
    new SaulAtLeast[T, U](coll.map(sensors), k)
  }
}

sealed trait SaulConstraint[T] {
  def and4[U](cons: SaulConstraint[U]) = {
    new SaulPairConjunction[T, U](this, cons)
  }

  def or4[U](cons: SaulConstraint[U]) = {
    new SaulPairDisjunction[T, U](this, cons)
  }

  def implies[U](q: SaulConstraint[U]): SaulImplication[T, U] = {
    new SaulImplication[T, U](this, q)
  }

  def ====>[U](q: SaulConstraint[U]): SaulImplication[T, U] = implies(q)

  def negate: SaulNegation[T] = {
    new SaulNegation(this)
  }

  def unary_! = negate
}

// zero-th order constraints
sealed trait SaulPropositionalConstraint[T] extends SaulConstraint[T] {
  def estimator: LBJLearnerEquivalent
}

case class SaulPropositionalEqualityConstraint[T](
  estimator: LBJLearnerEquivalent,
  instanceOpt: Option[T],
  equalityValOpt: Option[String],
  inequalityValOpt: Option[String]
) extends SaulPropositionalConstraint[T] {
  def is2(targetValue: String): SaulPropositionalEqualityConstraint[T] = new SaulPropositionalEqualityConstraint[T](estimator, instanceOpt, Some(targetValue), None)
  def isTrue2 = is2("true")
  def isFalse2 = is2("false")
  def isNot2(targetValue: String): SaulPropositionalEqualityConstraint[T] = new SaulPropositionalEqualityConstraint[T](estimator, instanceOpt, None, Some(targetValue))
}

case class SaulPairConjunction[T, U](c1: SaulConstraint[T], c2: SaulConstraint[U]) extends SaulConstraint[T]

case class SaulPairDisjunction[T, U](c1: SaulConstraint[T], c2: SaulConstraint[U]) extends SaulConstraint[T]

case class SaulForAll[T, U](constraints: Seq[SaulConstraint[U]]) extends SaulConstraint[T]

case class SaulExists[T, U](constraints: Seq[SaulConstraint[U]]) extends SaulConstraint[T]

case class SaulAtLeast[T, U](constraints: Seq[SaulConstraint[U]], k: Int) extends SaulConstraint[T]

case class SaulAtMost[T, U](constraints: Seq[SaulConstraint[U]], k: Int) extends SaulConstraint[T]

case class SaulImplication[T, U](p: SaulConstraint[T], q: SaulConstraint[U]) extends SaulConstraint[T]

case class SaulNegation[T](p: SaulConstraint[T]) extends SaulConstraint[T]