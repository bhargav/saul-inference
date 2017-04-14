/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling

import edu.illinois.cs.cogcomp.annotation.{Annotator, AnnotatorConfigurator, AnnotatorException}
import edu.illinois.cs.cogcomp.core.datastructures.{IntPair, ViewNames}
import edu.illinois.cs.cogcomp.core.datastructures.textannotation._
import edu.illinois.cs.cogcomp.core.utilities.configuration.ResourceManager
import edu.illinois.cs.cogcomp.edison.annotators.ClauseViewGenerator
import edu.illinois.cs.cogcomp.saul.classifier.ClassifierUtils
import edu.illinois.cs.cogcomp.saulexamples.nlp.CommonSensors

import scala.collection.JavaConversions._

class SRLAnnotatorConfigurator extends AnnotatorConfigurator

class SRLAnnotator(finalViewName: String = ViewNames.SRL_VERB, resourceManager: ResourceManager = new SRLAnnotatorConfigurator().getDefaultConfig)
  extends Annotator(finalViewName, SRLAnnotator.requiredViews, resourceManager) {
  val requiredViewSet: Set[String] = getRequiredViews.toSet
  lazy val clauseViewGenerator = ClauseViewGenerator.STANFORD

  override def addView(ta: TextAnnotation): Unit = {
    checkPrequisites(ta)

    SRLApps.srlDataModelObject.clearInstances()

    val finalView = new PredicateArgumentView(getViewName, SRLAnnotator.getClass.getCanonicalName, ta, 1.0)

    // Get Predicates in the sentence.
    val predicates = getPredicates(ta)

    predicates.foreach({ predicate: Constituent =>

      // Get arguments for each predicate detected.
      val argumentList = getArguments(ta, predicate)
      finalView.addPredicateArguments(
        predicate,
        argumentList.map(_.getTarget).toList,
        argumentList.map(_.getRelationName).toArray,
        argumentList.map(_.getScore).toArray)
    })

    assert(finalView.getConstituents.forall(_.getViewName == getViewName), "Verify correct constituent view names.")
    ta.addView(getViewName, finalView)

    SRLApps.srlDataModelObject.clearInstances()
  }

  override def initialize(rm: ResourceManager): Unit = {
    // Load models and other things
    ClassifierUtils.LoadClassifier(SRLConfigurator.SRL_JAR_MODEL_PATH.value + "/models_dTr/",
      SRLClassifiers.predicateClassifier)
    ClassifierUtils.LoadClassifier(SRLConfigurator.SRL_JAR_MODEL_PATH.value + "/models_bTr/",
      SRLClassifiers.argumentXuIdentifierGivenApredicate)
    ClassifierUtils.LoadClassifier(SRLConfigurator.SRL_JAR_MODEL_PATH.value + "/models_aTr/",
      SRLClassifiers.argumentTypeLearner)
  }

  def checkPrequisites(ta: TextAnnotation): Unit = {
    val missingRequirements = requiredViewSet.diff(ta.getAvailableViews)
    if (missingRequirements.nonEmpty) {
      throw new AnnotatorException(s"Document ${ta.getId} is missing required views: $missingRequirements")
    }

    clauseViewGenerator.addView(ta)
    assert(ta.hasView(ViewNames.CLAUSES_STANFORD))
  }

  /**
    * @param ta Input Text Annotation instance.
    * @return Constituents that are not attached to any view yet.
    */
  private def getPredicates(ta: TextAnnotation): Iterable[Constituent] = {
    SRLApps.srlDataModelObject.clearInstances()

    SRLApps.srlDataModelObject.sentences.populate(Seq(ta), train = false, populateEdge = false)
    SRLApps.srlDataModelObject.tokens.populate(CommonSensors.textAnnotationToTokens(ta), train = false, populateEdge = false)
    SRLApps.srlDataModelObject.stringTree.populate(Seq(SRLSensors.textAnnotationToStringTree(ta)), train = false, populateEdge = false)

    val predicateCandidates = ta.getView(ViewNames.TOKENS).map(_.cloneForNewView(getViewName))
    SRLApps.srlDataModelObject.predicates.populate(predicateCandidates, train = false, populateEdge = false)

    // Figure out the constants in Boolean Property
    // TODO - Constant for Predicate label
    predicateCandidates.filter(SRLClassifiers.predicateClassifier(_) == "true").map({ candidate: Constituent =>
      candidate.cloneForNewViewWithDestinationLabel(getViewName, "Predicate")
    })
  }

  /**
    * @param ta Input Text Annotation instance.
    * @param predicate Input Predicate instance.
    * @return Relation between unattached predicate and arguments.
    */
  private def getArguments(ta: TextAnnotation, predicate: Constituent): Iterable[Relation] = {
    SRLApps.srlDataModelObject.clearInstances()

    val stringTree = SRLSensors.textAnnotationToStringTree(ta)

    // Prevent duplicate clearing of graphs.
    SRLApps.srlDataModelObject.sentences.populate(Seq(ta), train = false, populateEdge = false)
    SRLApps.srlDataModelObject.tokens.populate(CommonSensors.textAnnotationToTokens(ta), train = false, populateEdge = false)
    SRLApps.srlDataModelObject.stringTree.populate(Seq(stringTree), train = false, populateEdge = false)
    SRLApps.srlDataModelObject.predicates.populate(Seq(predicate), train = false, populateEdge = false)

    val candidateRelations = SRLSensors.xuPalmerCandidate(predicate, stringTree)
    SRLApps.srlDataModelObject.arguments.populate(candidateRelations.map(_.getTarget), train = false, populateEdge = false)
    SRLApps.srlDataModelObject.relations.populate(candidateRelations, train = false, populateEdge = false)

    val finalRelationList = candidateRelations.filter({ candidate: Relation =>
      SRLClassifiers.argumentXuIdentifierGivenApredicate(candidate) == "true"
    })

    SRLApps.srlDataModelObject.arguments.clear()
    SRLApps.srlDataModelObject.arguments.populate(finalRelationList.map(_.getTarget), train = false, populateEdge = false)

    SRLApps.srlDataModelObject.relations.clear()
    SRLApps.srlDataModelObject.relations.populate(finalRelationList, train = false, populateEdge = false)

    finalRelationList.flatMap { relation: Relation =>
      val label = SRLClassifiers.argumentTypeLearner(relation)
      if (label == "candidate") None else Some(SRLAnnotator.cloneRelationWithNewLabel(relation, label))
    }
  }
}

object SRLAnnotator {
  private val requiredViews = Array(ViewNames.POS, ViewNames.LEMMA, ViewNames.SHALLOW_PARSE, ViewNames.PARSE_STANFORD)

  private def cloneRelationWithNewLabel(sourceRelation: Relation, label: String): Relation = {
    val newRelation = new Relation(label, sourceRelation.getSource, sourceRelation.getTarget, sourceRelation.getScore)
    sourceRelation.getAttributeKeys.foreach({ key: String =>
      newRelation.setAttribute(key, sourceRelation.getAttribute(key))
    })
    newRelation
  }
}
