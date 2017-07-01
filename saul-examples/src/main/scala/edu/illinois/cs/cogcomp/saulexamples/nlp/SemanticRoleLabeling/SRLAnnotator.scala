/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling

import edu.illinois.cs.cogcomp.annotation.{ Annotator, AnnotatorConfigurator, AnnotatorException }
import edu.illinois.cs.cogcomp.core.datastructures.ViewNames
import edu.illinois.cs.cogcomp.core.datastructures.textannotation._
import edu.illinois.cs.cogcomp.core.utilities.configuration.{ Configurator, Property, ResourceManager }
import edu.illinois.cs.cogcomp.edison.annotators.ClauseViewGenerator
import edu.illinois.cs.cogcomp.nlp.corpusreaders.AbstractSRLAnnotationReader
import edu.illinois.cs.cogcomp.saul.classifier.ClassifierUtils
import edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling.SRLClassifiers.SRLDataModel

import scala.collection.JavaConverters._

class SRLAnnotatorConfigurator extends AnnotatorConfigurator {
  override def getDefaultConfig: ResourceManager = {
    val props = Array[Property](
      SRLAnnotatorConfigurator.USE_PREDICATE_CLASSIFIER,
      SRLAnnotatorConfigurator.USE_ARGUMENT_IDENTIFIER,
      SRLAnnotatorConfigurator.USE_VERB_SENSE_CLASSIFIER,
      SRLAnnotatorConfigurator.USE_CONSTRAINTS,
      SRLAnnotatorConfigurator.USE_GREEDY_INFERENCE_BINARY,
      SRLAnnotatorConfigurator.USE_GREEDY_INFERENCE_TYPE
    )

    val defaultRm = super.getDefaultConfig
    Configurator.mergeProperties(defaultRm, new ResourceManager(generateProperties(props)))
  }
}

object SRLAnnotatorConfigurator {
  // Use the predicate classifier if true else use all verbs as predicates
  val USE_PREDICATE_CLASSIFIER = new Property("usePredicateClassifier", Configurator.TRUE)

  // Boolean denoting if we should perform Binary Argument Identification
  val USE_ARGUMENT_IDENTIFIER = new Property("useArgumentIdentifier", Configurator.TRUE)

  // Verb Sense Classifier is not trained currently
  val USE_VERB_SENSE_CLASSIFIER = new Property("useVerbSenseClassifier", Configurator.FALSE)

  // Constrained Inference
  val USE_CONSTRAINTS = new Property("useConstraints", Configurator.TRUE)

  // Use Greedy Inference at the Binary Argument Identifier
  val USE_GREEDY_INFERENCE_BINARY = new Property("useGreedyInferenceBinary", Configurator.FALSE)

  // Use Greedy Inference at the Argument Type Identifier
  val USE_GREEDY_INFERENCE_TYPE = new Property("useGreedyInferenceType", Configurator.FALSE)
}

class SRLAnnotator(finalViewName: String = ViewNames.SRL_VERB, resourceManager: ResourceManager = new SRLAnnotatorConfigurator().getDefaultConfig)
  extends Annotator(finalViewName, SRLAnnotator.requiredViews, resourceManager) {
  val requiredViewSet: Set[String] = getRequiredViews.toSet

  lazy val clauseViewGenerator: ClauseViewGenerator = {
    SRLscalaConfigurator.SRL_PARSE_VIEW match {
      case ViewNames.PARSE_GOLD => new ClauseViewGenerator(ViewNames.PARSE_GOLD, "CLAUSES_GOLD")
      case ViewNames.PARSE_STANFORD => ClauseViewGenerator.STANFORD
    }
  }

  override def addView(ta: TextAnnotation): Unit = {
    checkPrerequisites(ta)

    SRLDataModel.clearInstances()

    val finalView = new PredicateArgumentView(getViewName, SRLAnnotator.getClass.getCanonicalName, ta, 1.0)

    // Get Predicates in the sentence.
    val allPredicates = getPredicates(ta)

    allPredicates.foreach({ predicate: Constituent =>

      // Get arguments for each predicate detected.
      val argumentList = getArguments(ta, predicate)
      finalView.addPredicateArguments(
        predicate,
        argumentList.map(_.getTarget).toList.asJava,
        argumentList.map(_.getRelationName).toArray,
        argumentList.map(_.getScore).toArray
      )

      // Add additional attributes
      val lemmaOrToken = ta.getView(ViewNames.LEMMA)
        .getConstituentsCovering(predicate)
        .asScala
        .headOption
        .orElse(ta.getView(ViewNames.TOKENS).getConstituentsCovering(predicate).asScala.headOption)

      predicate.addAttribute(AbstractSRLAnnotationReader.SenseIdentifier, "XX")
      predicate.addAttribute(AbstractSRLAnnotationReader.LemmaIdentifier, lemmaOrToken.map(_.getLabel).getOrElse(""))
    })

    assert(finalView.getConstituents.asScala.forall(_.getViewName == getViewName), "Verify correct constituent view names.")
    ta.addView(getViewName, finalView)

    SRLDataModel.clearInstances()
  }

  override def initialize(rm: ResourceManager): Unit = {
    // Load models and other things
    ClassifierUtils.LoadClassifier(
      SRLscalaConfigurator.SRL_JAR_MODEL_PATH + "/models_dTr/",
      SRLClassifiers.predicateClassifier
    )
    ClassifierUtils.LoadClassifier(
      SRLscalaConfigurator.SRL_JAR_MODEL_PATH + "/models_bTr/",
      SRLClassifiers.argumentXuIdentifierGivenApredicate
    )
    ClassifierUtils.LoadClassifier(
      SRLscalaConfigurator.SRL_JAR_MODEL_PATH + "/models_cTr/",
      SRLClassifiers.argumentTypeLearner
    )
  }

  def checkPrerequisites(ta: TextAnnotation): Unit = {
    val missingRequirements = requiredViewSet.diff(ta.getAvailableViews.asScala)
    if (missingRequirements.nonEmpty) {
      throw new AnnotatorException(s"Document ${ta.getId} is missing required views: $missingRequirements")
    }

    clauseViewGenerator.addView(ta)

    // Check if the Annotator Configuration is compatible
    val useConstraint = resourceManager.getBoolean(SRLAnnotatorConfigurator.USE_CONSTRAINTS)
    val useGreedyInferenceBinary = resourceManager.getBoolean(SRLAnnotatorConfigurator.USE_GREEDY_INFERENCE_BINARY)
    val useGreedyInferenceType = resourceManager.getBoolean(SRLAnnotatorConfigurator.USE_GREEDY_INFERENCE_TYPE)

    if (useConstraint && (useGreedyInferenceBinary || useGreedyInferenceType)) {
      new UnsupportedOperationException("Incompatible configuration")
    } else if (useGreedyInferenceBinary && (useConstraint || useGreedyInferenceType)) {
      new UnsupportedOperationException("Incompatible configuration")
    } else if (useGreedyInferenceType && (useConstraint || useGreedyInferenceBinary)) {
      new UnsupportedOperationException("Incompatible configuration")
    }
  }

  /** @param ta Input Text Annotation instance.
    * @return Constituents that are not attached to any view yet.
    */
  private def getPredicates(ta: TextAnnotation): Iterable[Constituent] = {
    // Filter only verbs as candidates to the predicate classifier
    val predicateCandidates = ta.getView(ViewNames.POS)
      .getConstituents
      .asScala
      .filter(_.getLabel.startsWith("VB"))
      .map(_.cloneForNewView(getViewName))

    if (resourceManager.getBoolean(SRLAnnotatorConfigurator.USE_PREDICATE_CLASSIFIER)) {
      SRLDataModel.clearInstances()
      SRLDataModel.predicates.populate(predicateCandidates, train = false)

      predicateCandidates.filter(SRLClassifiers.predicateClassifier(_) == "true").map({ candidate: Constituent =>
        candidate.cloneForNewViewWithDestinationLabel(getViewName, "Predicate")
      })
    } else {
      predicateCandidates
    }
  }

  /** @param ta Input Text Annotation instance.
    * @param predicate Input Predicate instance.
    * @return Relation between unattached predicate and arguments.
    */
  private def getArguments(ta: TextAnnotation, predicate: Constituent): Iterable[Relation] = {
    SRLDataModel.clearInstances()

    // Prevent duplicate clearing of graphs.
    SRLDataModel.sentences.populate(Seq(ta), train = false)

    val stringTree = (SRLDataModel.sentences(ta) ~> SRLDataModel.sentencesToStringTree).head

    val candidateRelations = SRLSensors.xuPalmerCandidate(predicate, stringTree)
    SRLDataModel.relations.populate(candidateRelations, train = false)

    val finalRelationList = {
      if (resourceManager.getBoolean(SRLAnnotatorConfigurator.USE_ARGUMENT_IDENTIFIER)) {
        val filteredCandidates = {
          if (resourceManager.getBoolean(SRLAnnotatorConfigurator.USE_GREEDY_INFERENCE_BINARY)) {
            val candidatesWithScores = candidateRelations.map({
              candidate => (candidate, SRLClassifiers.argumentXuIdentifierGivenApredicate.classifier.scores(candidate))
            })

            // Greedy No Overlap decode
            GreedyDecoder.decodeNoOverlap(candidatesWithScores, Set("false"))
              .filter(x => x._2.value == "true")
              .map(_._1)
          } else {
            candidateRelations.filter({ candidate: Relation =>
              SRLClassifiers.argumentXuIdentifierGivenApredicate(candidate) == "true"
            })
          }
        }

        // Re-create graph if the size of candidates are different after filtering
        if (filteredCandidates.size != candidateRelations.size) {
          SRLDataModel.clearInstances()

          // Prevent duplicate clearing of graphs.
          SRLDataModel.sentences.populate(Seq(ta), train = false)
          SRLDataModel.relations.populate(filteredCandidates, train = false)
        }

        filteredCandidates
      } else {
        candidateRelations
      }
    }

    if (resourceManager.getBoolean(SRLAnnotatorConfigurator.USE_CONSTRAINTS)) {
      finalRelationList.flatMap({ relation: Relation =>
        val label = SRLConstrainedClassifiers.argTypeConstraintClassifier(relation)
        if (label == "candidate")
          None
        else
          Some(SRLAnnotator.cloneRelationWithNewLabelAndArgument(relation, label, 1.0, getViewName))
      })
    } else {
      val relationWithScores = finalRelationList.map({ relation: Relation =>
        (relation, SRLClassifiers.argumentTypeLearner.classifier.scores(relation))
      })

      if (resourceManager.getBoolean(SRLAnnotatorConfigurator.USE_GREEDY_INFERENCE_TYPE)) {
        GreedyDecoder.decodeNoOverlap(relationWithScores, Set("candidate"))
          .filterNot(_._2.value == "candidate")
          .map({
            case (relation, score) =>
              SRLAnnotator.cloneRelationWithNewLabelAndArgument(relation, score.value, score.score, getViewName)
          })
      } else {
        relationWithScores.map({
          case (relation, scoreset) =>
            val label = scoreset.highScoreValue()
            (relation, scoreset.getScore(label))
        }).filterNot(_._2.value == "candidate")
          .map({
            case (relation, score) =>
              SRLAnnotator.cloneRelationWithNewLabelAndArgument(relation, score.value, score.score, getViewName)
          })
      }
    }
  }
}

object SRLAnnotator {
  private val requiredViews = Array(
    ViewNames.POS,
    ViewNames.LEMMA,
    ViewNames.SHALLOW_PARSE,
    SRLscalaConfigurator.SRL_PARSE_VIEW
  )

  private def cloneRelationWithNewLabelAndArgument(
    sourceRelation: Relation,
    label: String,
    score: Double,
    targetViewName: String
  ): Relation = {
    val newTargetConstituent = sourceRelation.getTarget.cloneForNewView(targetViewName)
    val newRelation = new Relation(label, sourceRelation.getSource, newTargetConstituent, score)
    sourceRelation.getAttributeKeys.asScala.foreach({ key: String =>
      newRelation.addAttribute(key, sourceRelation.getAttribute(key))
    })
    newRelation
  }
}
