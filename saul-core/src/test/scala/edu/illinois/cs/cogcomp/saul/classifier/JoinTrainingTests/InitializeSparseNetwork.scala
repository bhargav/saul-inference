/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.JoinTrainingTests

import edu.illinois.cs.cogcomp.lbjava.learn.{ LinearThresholdUnit, SparseNetworkLearner }
import edu.illinois.cs.cogcomp.saul.classifier.infer.ConstrainedClassifier
import edu.illinois.cs.cogcomp.saul.classifier.infer.solver.OJAlgo
import edu.illinois.cs.cogcomp.saul.classifier.{ ClassifierUtils, JointTrainSparseNetwork, Learnable }
import edu.illinois.cs.cogcomp.saul.datamodel.DataModel
import org.scalatest.{ FlatSpec, Matchers }

class InitializeSparseNetwork extends FlatSpec with Matchers {

  // Testing the original functions with real classifiers
  "integration test" should "work" in {
    // Initialize toy model
    import TestModel._
    object TestClassifier extends Learnable(tokens) {
      def label = testLabel
      override def feature = using(word)
      override lazy val classifier = new SparseNetworkLearner()
    }
    object TestClassifierWithExtendedFeatures extends Learnable(tokens) {
      def label = testLabel
      override def feature = using(word, biWord)
      override lazy val classifier = new SparseNetworkLearner()
    }
    object TestConstraintClassifier extends ConstrainedClassifier[String, String] {
      override def subjectTo = None
      override val solverType = OJAlgo
      override lazy val onClassifier = TestClassifier
    }

    object TestConstraintClassifierWithExtendedFeatures extends ConstrainedClassifier[String, String] {
      override def subjectTo = None
      override val solverType = OJAlgo
      override lazy val onClassifier = TestClassifierWithExtendedFeatures
    }

    val words = List("this", "is", "a", "test", "candidate", ".")
    tokens.populate(words)

    val cls = List(TestConstraintClassifier, TestConstraintClassifierWithExtendedFeatures)

    TestConstraintClassifier.onClassifier.classifier.getLexicon.size() should be(0)
    TestConstraintClassifierWithExtendedFeatures.onClassifier.classifier.getLexicon.size() should be(0)
    TestConstraintClassifier.onClassifier.classifier.getLabelLexicon.size() should be(0)
    TestConstraintClassifierWithExtendedFeatures.onClassifier.classifier.getLabelLexicon.size() should be(0)

    val clNet1 = TestConstraintClassifier.onClassifier.classifier.asInstanceOf[SparseNetworkLearner]
    val clNet2 = TestConstraintClassifierWithExtendedFeatures.onClassifier.classifier.asInstanceOf[SparseNetworkLearner]

    clNet1.getNetwork.size() should be(0)
    clNet2.getNetwork.size() should be(0)

    ClassifierUtils.InitializeClassifiers(tokens, cls: _*)

    TestConstraintClassifier.onClassifier.classifier.getLexicon.size() should be(6)
    TestConstraintClassifierWithExtendedFeatures.onClassifier.classifier.getLexicon.size() should be(12)
    TestConstraintClassifier.onClassifier.classifier.getLabelLexicon.size() should be(2)
    TestConstraintClassifierWithExtendedFeatures.onClassifier.classifier.getLabelLexicon.size() should be(2)

    clNet1.getNetwork.size() should be(2)
    clNet2.getNetwork.size() should be(2)

    val wv1 = clNet1.getNetwork.get(0).asInstanceOf[LinearThresholdUnit].getWeightVector
    val wv2 = clNet2.getNetwork.get(0).asInstanceOf[LinearThresholdUnit].getWeightVector

    wv1.size() should be(0)
    wv2.size() should be(0)
    TestClassifierWithExtendedFeatures.learn(2)
    JointTrainSparseNetwork.train(tokens, cls, 5, init = false)

    val wv1After = clNet1.getNetwork.get(0).asInstanceOf[LinearThresholdUnit].getWeightVector
    val wv2After = clNet2.getNetwork.get(0).asInstanceOf[LinearThresholdUnit].getWeightVector

    wv1After.size() should be(6)
    wv2After.size() should be(12)
  }

  object TestModel extends DataModel {
    val tokens = node[String]
    val iEdge = edge(tokens, tokens)
    val testLabel = property(tokens) { x: String => x.equals("candidate") }
    val word = property(tokens) { x: String => x }
    val biWord = property(tokens) { x: String => x + "-" + x }
  }
}
