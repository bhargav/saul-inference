/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling

import edu.illinois.cs.cogcomp.core.datastructures.ViewNames
import edu.illinois.cs.cogcomp.saul.classifier.ClassifierUtils
import edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling.SRLClassifiers._
import edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling.SRLConstrainedClassifiers.ArgTypeConstrainedClassifier
import org.scalatest.{ FlatSpec, Matchers }

class ModelsTest extends FlatSpec with Matchers {

  val viewsToAdd = Array(
    ViewNames.LEMMA, ViewNames.POS, ViewNames.SHALLOW_PARSE,
    ViewNames.PARSE_GOLD, ViewNames.SRL_VERB
  )

  SRLDataModel.clearInstances()
  PopulateSRLDataModel(testOnly = true, useGoldPredicate = true, useGoldArgBoundaries = true, usePipelineCaching = false)

  "argument type classifier (aTr)" should "work." in {
    ClassifierUtils.LoadClassifier(SRLscalaConfigurator.SRL_JAR_MODEL_PATH + "/models_aTr/", argumentTypeLearner)
    val results = argumentTypeLearner.test(exclude = "candidate")
    results.perLabel
      .filter(!_.f1.isNaN)
      .foreach {
        result =>
          result.label match {
            case "A0" => result.f1 should be(0.95 +- 0.03)
            case "A1" => result.f1 should be(0.95 +- 0.03)
            case "A2" => result.f1 should be(0.85 +- 0.03)
            case _ => (result.f1 >= 0.0) should be(true)
          }
      }
  }

  "predicate identifier (dTr)" should "perform higher than 0.98." in {
    ClassifierUtils.LoadClassifier(SRLscalaConfigurator.SRL_JAR_MODEL_PATH + "/models_dTr/", predicateClassifier)
    val results = predicateClassifier.test()
    results.perLabel.foreach {
      result =>
        result.label match { case "true" => result.f1 should be(0.99 +- 0.01) }
    }
  }

  //TODO fix the issue with OjAlgo, and un-ignore this
  "L+I argument type classifier (cTr)" should "work." ignore {
    ClassifierUtils.LoadClassifier(SRLscalaConfigurator.SRL_JAR_MODEL_PATH + "/models_cTr/", argumentTypeLearner)
    val scores = ArgTypeConstrainedClassifier.test(exclude = "candidate")
    scores.perLabel.foreach { resultPerLabel =>
      resultPerLabel.label match {
        case "A0" => resultPerLabel.f1 should be(0.96 +- 0.02)
        case "A1" => resultPerLabel.f1 should be(0.93 +- 0.02)
        case "A2" => resultPerLabel.f1 should be(0.85 +- 0.02)
        case _ => ""
      }
    }
  }

  "argument identifier (bTr)" should "perform higher than 0.95." in {
    ClassifierUtils.LoadClassifier(SRLscalaConfigurator.SRL_JAR_MODEL_PATH + "/models_bTr/", argumentXuIdentifierGivenApredicate)
    val results = argumentXuIdentifierGivenApredicate.test()
    results.perLabel.foreach {
      result =>
        result.label match { case "true" => (result.f1 >= 0.95) should be(true) }
    }
  }

  "argument identifier (cTr) trained with XuPalmer" should "perform higher than 0.9." in {
    ClassifierUtils.LoadClassifier(SRLscalaConfigurator.SRL_JAR_MODEL_PATH + "/models_cTr/", argumentTypeLearner)
    val results = argumentTypeLearner.test()
    results.perLabel.foreach {
      result =>
        result.label match {
          case "A0" => result.f1 should be(0.95 +- 0.05)
          case "A1" => result.f1 should be(0.95 +- 0.05)
          case "A2" => result.f1 should be(0.85 +- 0.05)
          case _ => ""
        }
    }
  }

  "argument identifier (fTr) trained with XuPalmer and candidate predicates" should "work." in {
    ClassifierUtils.LoadClassifier(SRLscalaConfigurator.SRL_JAR_MODEL_PATH + "/models_fTr/", argumentTypeLearner)
    val results = argumentTypeLearner.test(exclude = "candidate")
    results.perLabel.foreach {
      result =>
        result.label match {
          case "A0" => result.f1 should be(0.95 +- 0.03)
          case "A1" => result.f1 should be(0.95 +- 0.03)
          case "A2" => result.f1 should be(0.85 +- 0.03)
          case _ => ""
        }
    }
  }
}

