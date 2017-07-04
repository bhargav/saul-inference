/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.infer

import java.io.PrintStream

import edu.illinois.cs.cogcomp.lbjava.classify.{ FeatureVector, ScoreSet }
import edu.illinois.cs.cogcomp.lbjava.learn.Learner
import edu.illinois.cs.cogcomp.saul.classifier.infer.Constraint._
import edu.illinois.cs.cogcomp.saul.classifier.infer.solver.OJAlgo
import edu.illinois.cs.cogcomp.saul.classifier.infer.{ ConstrainedClassifier, Constraint }
import edu.illinois.cs.cogcomp.saul.datamodel.DataModel
import edu.illinois.cs.cogcomp.saul.lbjrelated.LBJLearnerEquivalent
import org.scalatest.{ FlatSpec, Matchers }

class StaticClassifier(trueLabelScore: Double) extends Learner("DummyClassifer") {
  override def getInputType: String = { "DummyInstance" }

  override def allowableValues: Array[String] = { Array[String]("false", "true") }

  override def equals(o: Any): Boolean = { getClass == o.getClass }

  /** The reason for true to be -1 is because the internal optimization by default finds the maximizer, while in this
    * problem we are looking for a minimizer
    */
  override def scores(example: AnyRef): ScoreSet = {
    val result: ScoreSet = new ScoreSet
    result.put("false", 0)
    result.put("true", trueLabelScore)
    result
  }

  override def write(printStream: PrintStream): Unit = ???

  override def scores(ints: Array[Int], doubles: Array[Double]): ScoreSet = ???

  override def classify(ints: Array[Int], doubles: Array[Double]): FeatureVector = ???

  override def learn(ints: Array[Int], doubles: Array[Double], ints1: Array[Int], doubles1: Array[Double]): Unit = ???
}

case class Instance(value: Int)

object DummyDataModel extends DataModel {
  val instanceNode = node[Instance]

  // only used for implications
  val instanceNode2 = node[Instance]

  /** definition of the constraints */
  val classifierPositiveScoreForTrue: LBJLearnerEquivalent = new LBJLearnerEquivalent {
    override val classifier: Learner = new StaticClassifier(1.0)
  }
  val classifierNegativeScoreForTrue: LBJLearnerEquivalent = new LBJLearnerEquivalent {
    override val classifier: Learner = new StaticClassifier(-1.0)
  }

  def singleInstanceMustBeTrue(x: Instance) = { classifierNegativeScoreForTrue on x isTrue }
  def singleInstanceMustBeFalse(x: Instance) = { classifierPositiveScoreForTrue on x isFalse }
  def forAllTrue = instanceNode.ForAll { x: Instance => classifierPositiveScoreForTrue on x isTrue }
  def forAllFalse = instanceNode.ForAll { x: Instance => classifierPositiveScoreForTrue on x isFalse }
  def forAllOneOfTheLabelsPositiveClassifier = instanceNode.ForAll { x: Instance => classifierPositiveScoreForTrue on x isOneOf ("true", "true") }
  def forAllOneOfTheLabelsNegativeClassifier = instanceNode.ForAll { x: Instance => classifierPositiveScoreForTrue on x isOneOf ("true", "true") }
  def forAllNotFalse = instanceNode.ForAll { x: Instance => classifierPositiveScoreForTrue on x isNot "false" }
  def forAllNotTrue = instanceNode.ForAll { x: Instance => classifierPositiveScoreForTrue on x isNot "true" }
  def existsTrue = instanceNode.Exists { x: Instance => classifierNegativeScoreForTrue on x isTrue }
  def existsFalse = instanceNode.Exists { x: Instance => classifierPositiveScoreForTrue on x isFalse }
  def exatclyTrue(k: Int) = instanceNode.Exactly(k) { x: Instance => classifierPositiveScoreForTrue on x isTrue }
  def exatclyFalse(k: Int) = instanceNode.Exactly(k) { x: Instance => classifierPositiveScoreForTrue on x isFalse }
  def atLeastTrue(k: Int) = instanceNode.AtLeast(k) { x: Instance => classifierNegativeScoreForTrue on x isTrue }
  def atLeastFalse(k: Int) = instanceNode.AtLeast(k) { x: Instance => classifierPositiveScoreForTrue on x isFalse }
  def atMostTrue(k: Int) = instanceNode.AtMost(k) { x: Instance => classifierPositiveScoreForTrue on x isTrue }
  def atMostFalse(k: Int) = instanceNode.AtMost(k) { x: Instance => classifierNegativeScoreForTrue on x isFalse }
  def classifierHasSameValueOnTwoInstances(x: Instance, y: Instance) = classifierPositiveScoreForTrue on x equalsTo y

  // negation
  def forAllFalseWithNegation = instanceNode.ForAll { x: Instance => !(classifierPositiveScoreForTrue on x isTrue) }
  def forAllTrueNegated = !forAllTrue
  def atLeastFalseNegated(k: Int) = !atLeastFalse(k)

  // conjunction
  def allTrueAllTrueConjunction = forAllTrue and forAllTrue
  def allTrueAllFalseConjunction = forAllTrue and forAllFalse
  def allFalseAllTrueConjunction = forAllFalse and forAllTrue
  def allFalseAllFalseConjunction = forAllFalse and forAllFalse

  // disjunction
  def allTrueAllTrueDisjunction = forAllTrue or forAllTrue
  def allTrueAllFalseDisjunction = forAllTrue or forAllFalse
  def allFalseAllTrueDisjunction = forAllFalse or forAllTrue
  def allFalseAllFalseDisjunction = forAllFalse or forAllFalse
}

class DummyConstrainedInference(someConstraint: Some[Constraint[Instance]], classifier: LBJLearnerEquivalent) extends ConstrainedClassifier[Instance, Instance] {
  override lazy val onClassifier = classifier
  override def pathToHead = None
  override def subjectTo = someConstraint
  override def solverType = OJAlgo
}

class InferenceTest extends FlatSpec with Matchers {
  import DummyDataModel._

  val instanceSet = (1 to 5).map(Instance)
  DummyDataModel.instanceNode.populate(instanceSet)

  // extra constraints based on data
  // all instances should have the same label
  def classifierHasSameValueOnTwoInstancesInstantiated = {
    classifierHasSameValueOnTwoInstances(instanceSet(0), instanceSet(1)) and
      classifierHasSameValueOnTwoInstances(instanceSet(1), instanceSet(2)) and
      classifierHasSameValueOnTwoInstances(instanceSet(2), instanceSet(3)) and
      classifierHasSameValueOnTwoInstances(instanceSet(3), instanceSet(4))
  }

  def allInstancesShouldBeTrue = {
    classifierHasSameValueOnTwoInstancesInstantiated and singleInstanceMustBeTrue(instanceSet(0))
  }

  def trueImpliesTrue = {
    ((classifierNegativeScoreForTrue on instanceSet(0) isTrue) ==>
      (classifierNegativeScoreForTrue on instanceSet(1) isTrue)) and (classifierNegativeScoreForTrue on instanceSet(0) isTrue)
  }

  def trueImpliesFalse = {
    ((classifierNegativeScoreForTrue on instanceSet(0) isTrue) ==>
      (classifierNegativeScoreForTrue on instanceSet(1) isFalse)) and (classifierNegativeScoreForTrue on instanceSet(0) isTrue)
  }

  def falseImpliesTrue = {
    ((classifierNegativeScoreForTrue on instanceSet(0) isFalse) ==>
      (classifierNegativeScoreForTrue on instanceSet(1) isTrue)) and (classifierNegativeScoreForTrue on instanceSet(0) isFalse)
  }

  def falseImpliesFalse = {
    ((classifierNegativeScoreForTrue on instanceSet(0) isFalse) ==>
      (classifierNegativeScoreForTrue on instanceSet(1) isFalse)) and (classifierNegativeScoreForTrue on instanceSet(0) isFalse)
  }

  def halfHalfConstraint(classifier: LBJLearnerEquivalent, firstHalfLabel: String, secondHalfLabel: String) = {
    (0 to instanceSet.size / 2).map(i => classifier on instanceSet(i) is firstHalfLabel).ForAll and
      ((instanceSet.size / 2 + 1) until instanceSet.size).map(i => classifier on instanceSet(i) is secondHalfLabel).ForAll
  }

  def conjunctionOfDisjunction = {
    (classifierPositiveScoreForTrue on instanceSet(0) isFalse) and (
      (classifierPositiveScoreForTrue on instanceSet(1) isFalse) or
      (classifierPositiveScoreForTrue on instanceSet(2) isFalse)
    )
  }

  def disjunctionOfConjunctions = {
    (classifierPositiveScoreForTrue on instanceSet(0) isFalse) or (
      (classifierPositiveScoreForTrue on instanceSet(1) isFalse) and
      (classifierPositiveScoreForTrue on instanceSet(2) isFalse)
    )
  }

  def halfTrueHalfFalsePositiveClassifier = {
    halfHalfConstraint(classifierPositiveScoreForTrue, "true", "false") or
      halfHalfConstraint(classifierPositiveScoreForTrue, "false", "true")
  }

  def halfTrueHalfFalseNegativeClassifier = {
    halfHalfConstraint(classifierNegativeScoreForTrue, "true", "false") or
      halfHalfConstraint(classifierNegativeScoreForTrue, "false", "true")
  }

  // single instance constraint
  // TODO - Investigate this test case for ILP
  "first instance " should "true and the rest should be false" ignore {
    val singleInstanceMustBeTrueInference = new DummyConstrainedInference(
      Some(singleInstanceMustBeTrue(instanceSet.head)), classifierNegativeScoreForTrue
    )
    singleInstanceMustBeTrueInference(instanceSet.head) should be("true")
    instanceSet.drop(1).foreach { ins => singleInstanceMustBeTrueInference(ins) should be("false") }
  }

  // single instance constraint
  // TODO - Investigate this test case for ILP
  "first instance " should "false and the rest should be true" ignore {
    val singleInstanceMustBeFalseInference = new DummyConstrainedInference(
      Some(singleInstanceMustBeFalse(instanceSet.head)), classifierPositiveScoreForTrue
    )
    singleInstanceMustBeFalseInference(instanceSet.head) should be("false")
    instanceSet.drop(1).foreach { ins => singleInstanceMustBeFalseInference(ins) should be("true") }
  }

  // all true
  "ForAllTrue " should " return all true instances" in {
    val allTrueInference = new DummyConstrainedInference(Some(forAllTrue), classifierPositiveScoreForTrue)
    instanceSet.foreach { ins => allTrueInference(ins) should be("true") }
  }

  // all false
  "ForAllFalse " should " return all false instances" in {
    val allFalseInference = new DummyConstrainedInference(Some(forAllFalse), classifierPositiveScoreForTrue)
    instanceSet.foreach { ins => allFalseInference(ins) should be("false") }
  }

  // for all one of the labels
  "OneOf(true, some label) with positive true weight " should " work properly " in {
    val forAllOneOfTheLabelsPositiveClassifierInference = new DummyConstrainedInference(
      Some(forAllOneOfTheLabelsPositiveClassifier), classifierPositiveScoreForTrue
    )
    instanceSet.foreach { ins => forAllOneOfTheLabelsPositiveClassifierInference(ins) should be("true") }
  }

  // for all one of the labels
  "OneOf(true, some label) with negative true weight " should " work properly " in {
    val forAllOneOfTheLabelsNegativeClassifierInference = new DummyConstrainedInference(
      Some(forAllOneOfTheLabelsNegativeClassifier), classifierPositiveScoreForTrue
    )
    instanceSet.foreach { ins => forAllOneOfTheLabelsNegativeClassifierInference(ins) should be("true") }
  }

  // all not false, should always return true
  "ForAllNotFalse " should " return all true instances" in {
    val allNotFalseInference = new DummyConstrainedInference(Some(forAllNotFalse), classifierPositiveScoreForTrue)
    instanceSet.foreach { ins => allNotFalseInference(ins) should be("true") }
  }

  // all not true, should always return false
  "ForAllNotTrue " should " return all false instances" in {
    val allNotTrueInference = new DummyConstrainedInference(Some(forAllNotTrue), classifierPositiveScoreForTrue)
    instanceSet.foreach { ins => allNotTrueInference(ins) should be("false") }
  }

  // exists true
  "ExistsTrue " should " return exactly one true when true weight is negative" in {
    val existOneTrueInference = new DummyConstrainedInference(Some(existsTrue), classifierNegativeScoreForTrue)
    instanceSet.count { ins => existOneTrueInference(ins) == "true" } should be(1)
  }

  // exists false
  "ExistsFalse " should " return exactly one false when true weight is positive" in {
    val existOneFalseInference = new DummyConstrainedInference(Some(existsFalse), classifierPositiveScoreForTrue)
    instanceSet.count { ins => existOneFalseInference(ins) == "false" } should be(1)
  }

  // at least 2 true
  "AtLeast2True " should " return at least two true instance" in {
    val atLeastTwoTrueInference = new DummyConstrainedInference(Some(atLeastTrue(2)), classifierNegativeScoreForTrue)
    instanceSet.count { ins => atLeastTwoTrueInference(ins) == "true" } should be(2)
  }

  // at least 2 false
  "AtLeast2False " should " return at least two false instance" in {
    val atLeastTwoFalseInference = new DummyConstrainedInference(Some(atLeastFalse(2)), classifierPositiveScoreForTrue)
    instanceSet.count { ins => atLeastTwoFalseInference(ins) == "false" } should be(2)
  }

  // at least 3 true
  "AtLeast3True " should " return at least three true instance" in {
    val atLeastThreeTrueInference = new DummyConstrainedInference(Some(atLeastTrue(3)), classifierNegativeScoreForTrue)
    instanceSet.count { ins => atLeastThreeTrueInference(ins) == "true" } should be(3)
  }

  // at least 3 false
  "AtLeast3False " should " return at least three false instance" in {
    val atLeastThreeFalseInference = new DummyConstrainedInference(Some(atLeastFalse(3)), classifierPositiveScoreForTrue)
    instanceSet.count { ins => atLeastThreeFalseInference(ins) == "false" } should be >= 3
  }

  // exactly 1 true
  "ExactlyOneTrue " should " return exactly one true instance" in {
    val exactlyOneTrue = new DummyConstrainedInference(Some(exatclyTrue(1)), classifierPositiveScoreForTrue)
    instanceSet.count { ins => exactlyOneTrue(ins) == "true" } should be(1)
  }

  // exactly 2 true
  "ExactlyTwoTrue " should " return exactly two true instances" in {
    val exactlyOneTrue = new DummyConstrainedInference(Some(exatclyTrue(2)), classifierPositiveScoreForTrue)
    instanceSet.count { ins => exactlyOneTrue(ins) == "true" } should be(2)
  }

  // exactly 3 true
  "ExactlyTwoTrue " should " return exactly three true instances" in {
    val exactlyOneTrue = new DummyConstrainedInference(Some(exatclyTrue(3)), classifierPositiveScoreForTrue)
    instanceSet.count { ins => exactlyOneTrue(ins) == "true" } should be(3)
  }

  // exactly 1 false
  "ExactlyOneFalse " should " return exactly one true instances" in {
    val exactlyOneFalse = new DummyConstrainedInference(Some(exatclyFalse(1)), classifierPositiveScoreForTrue)
    instanceSet.count { ins => exactlyOneFalse(ins) == "false" } should be(1)
  }

  // exactly 2 false
  "ExactlyTwoFalse " should " return exactly two true instances" in {
    val exactlyOneFalse = new DummyConstrainedInference(Some(exatclyFalse(2)), classifierPositiveScoreForTrue)
    instanceSet.count { ins => exactlyOneFalse(ins) == "false" } should be(2)
  }

  // exactly 3 false
  "ExactlyTwoFalse " should " return exactly three true instances" in {
    val exactlyOneFalse = new DummyConstrainedInference(Some(exatclyFalse(3)), classifierPositiveScoreForTrue)
    instanceSet.count { ins => exactlyOneFalse(ins) == "false" } should be(3)
  }

  // at most 2 true
  "AtMost " should " return at most two true instances" in {
    val atMostTwoTrueInference = new DummyConstrainedInference(Some(atMostTrue(1)), classifierPositiveScoreForTrue)
    instanceSet.count { ins => atMostTwoTrueInference(ins) == "true" } should be(1)
  }

  // at most 2 false
  "AtMost " should " return at most two false instances" in {
    val atMostTwoFalseInference = new DummyConstrainedInference(Some(atMostFalse(1)), classifierNegativeScoreForTrue)
    instanceSet.count { ins => atMostTwoFalseInference(ins) == "false" } should be(1)
  }

  // at most 3 true
  "AtMost " should " return at most three true instances" in {
    val atMostThreeTrueInference = new DummyConstrainedInference(Some(atMostTrue(3)), classifierPositiveScoreForTrue)
    instanceSet.count { ins => atMostThreeTrueInference(ins) == "true" } should be(3)
  }

  // at most 3 false
  "AtMost " should " return at most three false instances" in {
    val atMostThreeFalseInference = new DummyConstrainedInference(Some(atMostFalse(3)), classifierNegativeScoreForTrue)
    instanceSet.count { ins => atMostThreeFalseInference(ins) == "false" } should be(3)
  }

  // negation of ForAllTrue
  "ForAllFalseWithNegation " should " all be false" in {
    val forAllFalseWithNegationInference = new DummyConstrainedInference(Some(forAllFalseWithNegation), classifierPositiveScoreForTrue)
    instanceSet.count { ins => forAllFalseWithNegationInference(ins) == "false" } should be(instanceSet.length)
  }

  // negation of ForAllTrue
  "ForAllTrueNegated " should " contain at least one false" in {
    val forAllTrueNegatedInference = new DummyConstrainedInference(Some(forAllTrueNegated), classifierPositiveScoreForTrue)
    instanceSet.count { ins => forAllTrueNegatedInference(ins) == "false" } should be >= 1
  }

  // conjunctions
  "AllTrueAllTrueConjunction " should " always be true" in {
    val allTrueAllTrueConjunctionInference = new DummyConstrainedInference(Some(allTrueAllTrueConjunction), classifierPositiveScoreForTrue)
    instanceSet.forall { ins => allTrueAllTrueConjunctionInference(ins) == "true" } should be(true)
  }

  "AllFalseAllTrueConjunction " should " always be false" in {
    val allFalseAllFalseConjunctionInference = new DummyConstrainedInference(Some(allFalseAllFalseConjunction), classifierPositiveScoreForTrue)
    instanceSet.forall { ins => allFalseAllFalseConjunctionInference(ins) == "false" } should be(true)
  }

  // disjunctions
  "AllTrueAllTrueDisjunction " should " always be true" in {
    val allTrueAllTrueDisjunctionInference = new DummyConstrainedInference(Some(allTrueAllTrueDisjunction), classifierPositiveScoreForTrue)
    instanceSet.forall { ins => allTrueAllTrueDisjunctionInference(ins) == "true" } should be(true)
  }

  "AllFalseAllFalseDisjunction " should " always be false" in {
    val allFalseAllFalseDisjunctionInference = new DummyConstrainedInference(Some(allFalseAllFalseDisjunction), classifierPositiveScoreForTrue)
    instanceSet.count { ins => allFalseAllFalseDisjunctionInference(ins) == "false" } should be(instanceSet.size)
  }

  "AllTrueAllFalseDisjunction " should " always all be false, or should all be true" in {
    val allTrueAllFalseDisjunctionInference = new DummyConstrainedInference(Some(allTrueAllFalseDisjunction), classifierPositiveScoreForTrue)
    (instanceSet.forall { ins => allTrueAllFalseDisjunctionInference(ins) == "false" } ||
      instanceSet.forall { ins => allTrueAllFalseDisjunctionInference(ins) == "true" }) should be(true)
  }

  "AllFalseAllTrueDisjunction " should " always all be false, or should all be true" in {
    val allFalseAllTrueDisjunctionInference = new DummyConstrainedInference(Some(allFalseAllTrueDisjunction), classifierPositiveScoreForTrue)
    (instanceSet.forall { ins => allFalseAllTrueDisjunctionInference(ins) == "false" } ||
      instanceSet.forall { ins => allFalseAllTrueDisjunctionInference(ins) == "true" }) should be(true)
  }

  "classifiers with instance pair label equality constraint " should " have the same value for all instances" in {
    val classifierSameValueTwoInstancesInference = new DummyConstrainedInference(
      Some(allInstancesShouldBeTrue), classifierPositiveScoreForTrue
    )
    instanceSet.forall { ins => classifierSameValueTwoInstancesInference(ins) == "true" } should be(true)
  }

  "trueImpliesTrue " should "work" in {
    val classifierSameValueTwoInstancesInference = new DummyConstrainedInference(
      Some(trueImpliesTrue), classifierNegativeScoreForTrue
    )
    assert(classifierSameValueTwoInstancesInference(instanceSet(0)) == "true" &&
      classifierSameValueTwoInstancesInference(instanceSet(1)) == "true")
  }

  "trueImpliesFalse " should "work" in {
    val classifierSameValueTwoInstancesInference = new DummyConstrainedInference(
      Some(trueImpliesFalse), classifierNegativeScoreForTrue
    )
    assert(classifierSameValueTwoInstancesInference(instanceSet(0)) == "true" &&
      classifierSameValueTwoInstancesInference(instanceSet(1)) == "false")
  }

  "falseImpliesTrue " should "work" in {
    val classifierSameValueTwoInstancesInference = new DummyConstrainedInference(
      Some(falseImpliesTrue), classifierNegativeScoreForTrue
    )
    assert(classifierSameValueTwoInstancesInference(instanceSet(0)) == "false" &&
      classifierSameValueTwoInstancesInference(instanceSet(1)) == "true")
  }

  "falseImpliesFalse " should "work" in {
    val classifierSameValueTwoInstancesInference = new DummyConstrainedInference(
      Some(falseImpliesFalse), classifierNegativeScoreForTrue
    )
    assert(classifierSameValueTwoInstancesInference(instanceSet(0)) == "false" &&
      classifierSameValueTwoInstancesInference(instanceSet(1)) == "false")
  }

  "halfTrueHalfFalsePositiveClassifier" should " work properly" in {
    val halfTrueHalfFalsePositiveClassifierInference = new DummyConstrainedInference(
      Some(halfTrueHalfFalsePositiveClassifier), classifierPositiveScoreForTrue
    )
    assert(((0 to instanceSet.size / 2).forall(i => halfTrueHalfFalsePositiveClassifierInference(instanceSet(i)) == "true") &&
      ((instanceSet.size / 2 + 1) until instanceSet.size).forall(i => halfTrueHalfFalsePositiveClassifierInference(instanceSet(i)) == "false")) ||
      ((0 to instanceSet.size / 2).forall(i => halfTrueHalfFalsePositiveClassifierInference(instanceSet(i)) == "false") &&
        ((instanceSet.size / 2 + 1) until instanceSet.size).forall(i => halfTrueHalfFalsePositiveClassifierInference(instanceSet(i)) == "true")))
  }

  "conjunctionOfDisjunctions " should " work" in {
    val conjunctionOfDisjunctionInference = new DummyConstrainedInference(
      Some(conjunctionOfDisjunction), classifierPositiveScoreForTrue
    )
    (0 to 2).count { i =>
      conjunctionOfDisjunctionInference(instanceSet(i)) == "false"
    } should be(2)
  }

  "disjunctionOfConjunction " should " work" in {
    val disjunctionOfConjunctionsInference = new DummyConstrainedInference(
      Some(disjunctionOfConjunctions), classifierPositiveScoreForTrue
    )
    (0 to 2).count { i =>
      disjunctionOfConjunctionsInference(instanceSet(i)) == "false"
    } should be(1)
  }
}
