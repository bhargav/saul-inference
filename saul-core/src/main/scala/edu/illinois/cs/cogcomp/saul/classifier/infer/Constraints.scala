/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer

import edu.illinois.cs.cogcomp.saul.datamodel.node.Node
import edu.illinois.cs.cogcomp.saul.lbjrelated.LBJLearnerEquivalent

import scala.collection.mutable
import scala.reflect.ClassTag

import scala.collection.JavaConverters._
import scala.language.implicitConversions

object Constraint {
  implicit class LearnerToFirstOrderConstraint1(estimator: LBJLearnerEquivalent) {
    // connecting a classifier to a specific instance
    def on[T](newInstance: T)(implicit tag: ClassTag[T]): InstanceWrapper[T] = InstanceWrapper(newInstance, estimator)
  }

  implicit def toPropositionalEqualityConstraint[T](wrapper: InstanceWrapper[T])(
    implicit
    tag: ClassTag[T]
  ): PropositionalEqualityConstraint[T] = {
    new PropositionalEqualityConstraint[T](wrapper.estimator, Some(wrapper.instance), None, None)
  }
  implicit def toInstancePairEqualityConstraint[T](wrapper: InstanceWrapper[T])(implicit
    tag: ClassTag[T],
    d1: DummyImplicit): InstancePairEqualityConstraint[T] = {
    new InstancePairEqualityConstraint[T](wrapper.estimator, wrapper.instance, None, None)
  }
  implicit def toEstimatorPairEqualityConstraint[T](
    wrapper: InstanceWrapper[T]
  )(implicit tag: ClassTag[T]): EstimatorPairEqualityConstraint[T] = {
    new EstimatorPairEqualityConstraint[T](wrapper.estimator, None, wrapper.instance, None)
  }

  // collection of target object types
  implicit def firstOrderConstraint[T <: AnyRef](
    coll: Traversable[T]
  ): ConstraintObjWrapper[T] = new ConstraintObjWrapper[T](coll.toSeq)
  implicit def firstOrderConstraint[T <: AnyRef](
    coll: Set[T]
  ): ConstraintObjWrapper[T] = new ConstraintObjWrapper[T](coll.toSeq)
  implicit def firstOrderConstraint[T <: AnyRef](
    coll: java.util.Collection[T]
  ): ConstraintObjWrapper[T] = new ConstraintObjWrapper[T](coll.asScala.toSeq)
  implicit def firstOrderConstraint[T <: AnyRef](
    coll: mutable.LinkedHashSet[T]
  ): ConstraintObjWrapper[T] = new ConstraintObjWrapper[T](coll.toSeq)
  implicit def firstOrderConstraint[T <: AnyRef](
    node: Node[T]
  ): ConstraintObjWrapper[T] = {
    new ConstraintObjWrapper[T](node.getAllInstances.toSeq)
  }

  // node object
  implicit def nodeObjectConstraint[T <: AnyRef](node: Node[T]): NodeWrapper[T] = {
    new NodeWrapper[T](node)
  }

  // collection of constraints
  implicit def createConstraintCollection[T <: AnyRef](
    coll: Traversable[Constraint[T]]
  ): ConstraintCollection[T, T] = new ConstraintCollection[T, T](coll.toSet)
  implicit def createConstraintCollection[T <: AnyRef](
    coll: Set[Constraint[T]]
  ): ConstraintCollection[T, T] = new ConstraintCollection[T, T](coll)
  implicit def createConstraintCollection[T <: AnyRef](
    coll: java.util.Collection[Constraint[T]]
  ): ConstraintCollection[T, T] = new ConstraintCollection[T, T](coll.asScala.toSet)
  implicit def createConstraintCollection[T <: AnyRef](
    coll: mutable.LinkedHashSet[Constraint[T]]
  ): ConstraintCollection[T, T] = new ConstraintCollection[T, T](coll.toSet)
}

class ConstraintCollection[T, U](coll: Set[Constraint[U]]) {
  def ForAll = new ForAll[T, U](coll)
  def Exists = new AtLeast[T, U](coll, 1)
  def AtLeast(k: Int) = new AtLeast[T, U](coll, k)
  def AtMost(k: Int) = new AtMost[T, U](coll, k)
  def Exactly(k: Int) = new Exactly[T, U](coll, k)
}

class NodeWrapper[T <: AnyRef](node: Node[T]) {
  def ForEach(sensor: T => Constraint[_]) = PerInstanceConstraint(sensor)
}

case class PerInstanceConstraint[T](sensor: T => Constraint[_]) extends Constraint[T] {
  override def negate: Constraint[T] = ???
}

class ConstraintObjWrapper[T](coll: Seq[T]) {
  def ForAll[U](sensors: T => Constraint[U])(implicit tag: ClassTag[T]): ForAll[T, U] = {
    new ForAll[T, U](coll.map(sensors).toSet)
  }
  def Exists[U](sensors: T => Constraint[U])(implicit tag: ClassTag[T]): AtLeast[T, U] = {
    new AtLeast[T, U](coll.map(sensors).toSet, 1)
  }
  def AtLeast[U](k: Int)(sensors: T => Constraint[U])(implicit tag: ClassTag[T]): AtLeast[T, U] = {
    new AtLeast[T, U](coll.map(sensors).toSet, k)
  }
  def AtMost[U](k: Int)(sensors: T => Constraint[U])(implicit tag: ClassTag[T]): AtMost[T, U] = {
    new AtMost[T, U](coll.map(sensors).toSet, k)
  }
  def Exactly[U](k: Int)(sensors: T => Constraint[U])(implicit tag: ClassTag[T]): Exactly[T, U] = {
    new Exactly[T, U](coll.map(sensors).toSet, k)
  }
}

sealed trait Constraint[T] {
  def and[U](cons: Constraint[U]) = {
    new PairConjunction[T, U](this, cons)
  }

  def or[U](cons: Constraint[U]) = {
    new PairDisjunction[T, U](this, cons)
  }

  def implies[U](q: Constraint[U]): Implication[T, U] = {
    new Implication[T, U](this, q)
  }

  def ==>[U](q: Constraint[U]): Implication[T, U] = implies(q)

  def ifAndOnlyIf[U](q: Constraint[U]): PairDisjunction[T, T] = {
    // p <--> q can be modelled as (p and q) or (not(p) and not(q))
    PairConjunction[T, U](this, q) or PairConjunction[T, U](this.negate, q.negate)
  }
  def <==>[U](q: Constraint[U]): PairDisjunction[T, T] = ifAndOnlyIf(q)

  def negate: Constraint[T]
  def unary_! = negate
}

// zero-th order constraints
sealed trait PropositionalConstraint[T] extends Constraint[T] {
  def estimator: LBJLearnerEquivalent
}

case class InstanceWrapper[T](instance: T, estimator: LBJLearnerEquivalent)

case class PropositionalEqualityConstraint[T](
  estimator: LBJLearnerEquivalent,
  instanceOpt: Option[T],
  equalityValOpt: Option[String],
  inequalityValOpt: Option[String]
) extends PropositionalConstraint[T] {
  def is(targetValue: String): PropositionalEqualityConstraint[T] = new PropositionalEqualityConstraint[T](
    estimator, instanceOpt, Some(targetValue), None
  )
  def isTrue = is("true")
  def isFalse = is("false")
  def isNot(targetValue: String): PropositionalEqualityConstraint[T] = new PropositionalEqualityConstraint[T](
    estimator, instanceOpt, None, Some(targetValue)
  )
  def negate: Constraint[T] = {
    if (equalityValOpt.isDefined) {
      new PropositionalEqualityConstraint[T](estimator, instanceOpt, None, equalityValOpt)
    } else {
      new PropositionalEqualityConstraint[T](estimator, instanceOpt, inequalityValOpt, None)
    }
  }
  def isOneOf(values: Traversable[String]): AtLeast[T, T] = {
    val equalityConst = values.map { v =>
      new PropositionalEqualityConstraint[T](
        estimator, instanceOpt, Some(v), None
      )
    }
    new AtLeast[T, T](equalityConst.toSet, 1)
  }

  def isOneOf(values: String*): AtLeast[T, T] = {
    isOneOf(values.toArray)
  }
}

// the two estimators should have the same prediction on the given instance
case class EstimatorPairEqualityConstraint[T](
  estimator1: LBJLearnerEquivalent,
  estimator2Opt: Option[LBJLearnerEquivalent],
  instance: T,
  equalsOpt: Option[Boolean]
) extends PropositionalConstraint[T] {
  def equalsTo(estimator2: LBJLearnerEquivalent) = new EstimatorPairEqualityConstraint[T](
    this.estimator1,
    Some(estimator2),
    this.instance,
    Some(true)
  )
  def differentFrom(estimator2: LBJLearnerEquivalent) = new EstimatorPairEqualityConstraint[T](
    this.estimator1,
    Some(estimator2),
    this.instance,
    Some(false)
  )
  override def estimator: LBJLearnerEquivalent = ???
  override def negate: Constraint[T] = EstimatorPairEqualityConstraint(
    estimator1, estimator2Opt, instance, Some(!equalsOpt.get)
  )
}

case class InstancePairEqualityConstraint[T](
  estimator: LBJLearnerEquivalent,
  instance1: T,
  instance2Opt: Option[T],
  equalsOpt: Option[Boolean]
) extends PropositionalConstraint[T] {
  def equalsTo(instance2: T) = new InstancePairEqualityConstraint[T](
    this.estimator,
    this.instance1,
    Some(instance2),
    Some(true)
  )
  def differentFrom(instance2: T) = new InstancePairEqualityConstraint[T](
    this.estimator,
    this.instance1,
    Some(instance2),
    Some(false)
  )
  override def negate: Constraint[T] = InstancePairEqualityConstraint(
    estimator, instance1, instance2Opt, Some(!equalsOpt.get)
  )
}

case class PairConjunction[T, U](c1: Constraint[T], c2: Constraint[U]) extends Constraint[T] {
  def negate: Constraint[T] = new PairDisjunction[T, U](c1.negate, c2.negate)
}

case class PairDisjunction[T, U](c1: Constraint[T], c2: Constraint[U]) extends Constraint[T] {
  def negate: Constraint[T] = new PairConjunction[T, U](c1.negate, c2.negate)
}

sealed trait ConstraintCollections[T, U] extends Constraint[T] {
  val constraints: Set[Constraint[U]]
}

case class ForAll[T, U](constraints: Set[Constraint[U]]) extends ConstraintCollections[T, U] {
  def negate: Constraint[T] = new ForAll[T, U](constraints.map(_.negate))
}

case class AtLeast[T, U](constraints: Set[Constraint[U]], k: Int) extends ConstraintCollections[T, U] {
  def negate: Constraint[T] = new AtMost[T, U](constraints, k)
}

case class AtMost[T, U](constraints: Set[Constraint[U]], k: Int) extends ConstraintCollections[T, U] {
  def negate: Constraint[T] = new AtLeast[T, U](constraints, k)
}

case class Exactly[T, U](constraints: Set[Constraint[U]], k: Int) extends ConstraintCollections[T, U] {
  def negate: Constraint[T] = new Exactly[T, U](constraints.map(_.negate), k)
}

case class Implication[T, U](p: Constraint[T], q: Constraint[U]) extends Constraint[T] {
  def negate: Constraint[T] = Implication[T, U](p, q.negate)
}

case class Negation[T](p: Constraint[T]) extends Constraint[T] {
  // negation of negation
  def negate: Constraint[T] = p
}
