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
import edu.illinois.cs.cogcomp.core.utilities.configuration.ResourceManager
import edu.illinois.cs.cogcomp.edison.annotators.ClauseViewGenerator
import edu.illinois.cs.cogcomp.nlp.corpusreaders.AbstractSRLAnnotationReader
import edu.illinois.cs.cogcomp.saul.classifier.ClassifierUtils
import edu.illinois.cs.cogcomp.saulexamples.nlp.CommonSensors

import scala.collection.JavaConversions._

class SRLAnnotatorConfigurator extends AnnotatorConfigurator

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

    SRLMultiGraphDataModel.clearInstances()

    val finalView = new PredicateArgumentView(getViewName, SRLAnnotator.getClass.getCanonicalName, ta, 1.0)

    // Get Predicates in the sentence.
    val allPredicates = getPredicates(ta)

    allPredicates.foreach({ predicate: Constituent =>

      // Get arguments for each predicate detected.
      val argumentList = getArguments(ta, predicate)
      finalView.addPredicateArguments(
        predicate,
        argumentList.map(_.getTarget).toList,
        argumentList.map(_.getRelationName).toArray,
        argumentList.map(_.getScore).toArray
      )

      // Add additional attributes
      val lemmaOrToken = ta.getView(ViewNames.LEMMA)
        .getConstituentsCovering(predicate)
        .headOption
        .orElse(ta.getView(ViewNames.TOKENS).getConstituentsCovering(predicate).headOption)

      // TODO - Need to train a Predicate Sense identifier.
      predicate.addAttribute(AbstractSRLAnnotationReader.SenseIdentifier, "01")
      predicate.addAttribute(AbstractSRLAnnotationReader.LemmaIdentifier, lemmaOrToken.map(_.getLabel).getOrElse(""))
    })

    assert(finalView.getConstituents.forall(_.getViewName == getViewName), "Verify correct constituent view names.")
    ta.addView(getViewName, finalView)

    SRLMultiGraphDataModel.clearInstances()
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
    val missingRequirements = requiredViewSet.diff(ta.getAvailableViews)
    if (missingRequirements.nonEmpty) {
      throw new AnnotatorException(s"Document ${ta.getId} is missing required views: $missingRequirements")
    }

    clauseViewGenerator.addView(ta)
  }

  /** @param ta Input Text Annotation instance.
    * @return Constituents that are not attached to any view yet.
    */
  private def getPredicates(ta: TextAnnotation): Iterable[Constituent] = {
    SRLMultiGraphDataModel.clearInstances()

    SRLMultiGraphDataModel.sentences.populate(Seq(ta), train = false, populateEdge = false)
    SRLMultiGraphDataModel.tokens.populate(CommonSensors.textAnnotationToTokens(ta), train = false, populateEdge = false)
    SRLMultiGraphDataModel.stringTree.populate(Seq(SRLSensors.textAnnotationToStringTree(ta)), train = false, populateEdge = false)

    // Filter only verbs as candidates to the predicate classifier
    val predicateCandidates = ta.getView(ViewNames.POS)
      .getConstituents
      .filter(_.getLabel.startsWith("VB"))
      .map(_.cloneForNewView(getViewName))
    SRLMultiGraphDataModel.predicates.populate(predicateCandidates, train = false, populateEdge = false)

    predicateCandidates.filter(SRLClassifiers.predicateClassifier(_) == "true").map({ candidate: Constituent =>
      candidate.cloneForNewViewWithDestinationLabel(getViewName, "Predicate")
    })
  }

  /** @param ta Input Text Annotation instance.
    * @param predicate Input Predicate instance.
    * @return Relation between unattached predicate and arguments.
    */
  private def getArguments(ta: TextAnnotation, predicate: Constituent): Iterable[Relation] = {
    SRLMultiGraphDataModel.clearInstances()

    val stringTree = SRLSensors.textAnnotationToStringTree(ta)

    // Prevent duplicate clearing of graphs.
    SRLMultiGraphDataModel.sentences.populate(Seq(ta), train = false, populateEdge = false)
    SRLMultiGraphDataModel.tokens.populate(CommonSensors.textAnnotationToTokens(ta), train = false, populateEdge = false)
    SRLMultiGraphDataModel.stringTree.populate(Seq(stringTree), train = false, populateEdge = false)
    SRLMultiGraphDataModel.predicates.populate(Seq(predicate), train = false, populateEdge = false)

    val candidateRelations = SRLSensors.xuPalmerCandidate(predicate, stringTree)
    SRLMultiGraphDataModel.arguments.populate(candidateRelations.map(_.getTarget), train = false, populateEdge = false)
    SRLMultiGraphDataModel.relations.populate(candidateRelations, train = false, populateEdge = false)

    val finalRelationList = candidateRelations.filter({ candidate: Relation =>
      SRLClassifiers.argumentXuIdentifierGivenApredicate(candidate) == "true"
    })

    SRLMultiGraphDataModel.arguments.clear()
    SRLMultiGraphDataModel.arguments.populate(finalRelationList.map(_.getTarget), train = false, populateEdge = false)

    SRLMultiGraphDataModel.relations.clear()
    SRLMultiGraphDataModel.relations.populate(finalRelationList, train = false, populateEdge = false)

    finalRelationList.flatMap({ relation: Relation =>
      val label = SRLConstrainedClassifiers.argTypeConstraintClassifier(relation)
      if (label == "candidate")
        None
      else
        Some(SRLAnnotator.cloneRelationWithNewLabelAndArgument(relation, label, getViewName))
    })
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
    targetViewName: String
  ): Relation = {
    val newTargetConstituent = sourceRelation.getTarget.cloneForNewView(targetViewName)
    val newRelation = new Relation(label, sourceRelation.getSource, newTargetConstituent, sourceRelation.getScore)
    sourceRelation.getAttributeKeys.foreach({ key: String =>
      newRelation.addAttribute(key, sourceRelation.getAttribute(key))
    })
    newRelation
  }
}
