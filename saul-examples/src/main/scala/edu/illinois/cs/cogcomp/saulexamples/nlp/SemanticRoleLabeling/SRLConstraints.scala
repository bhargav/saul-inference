/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling

import edu.illinois.cs.cogcomp.core.datastructures.textannotation._
import edu.illinois.cs.cogcomp.saul.classifier.infer.Constraint._
import edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling.SRLApps.srlDataModelObject._
import edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling.SRLClassifiers.{ argumentTypeLearner, argumentXuIdentifierGivenApredicate }

object SRLConstraints {
  def argumentTypeConstraints = sentences.ForEach { x: TextAnnotation =>
    noOverlapConstraint.sensor(x) and legalArgumentsConstraint.sensor(x) and noDuplicatesConstraint.sensor(x) and
      continuationConstraint.sensor(x) and referenceConstraint.sensor(x)
  }

  def noOverlapConstraint = sentences.ForEach { x: TextAnnotation =>
    // Each TextAnnotation is a sentence here.
    (sentences(x) ~> sentencesToRelations ~> relationsToPredicates).toList.distinct.ForAll { verb: Constituent =>
      val relationsList = (predicates(verb) ~> -relationsToPredicates).toList.distinct
      (sentences(x) ~> sentencesToTokens)
        .filter(_ != verb)
        .ForAll { word: Constituent =>
          // Filter relations to include ones with the argument covering the current word.
          relationsList
            .filter({ p: Relation =>
              (relations(p) ~> relationsToArguments).toList.head.doesConstituentCover(word)
            })
            .AtMost(1) { p: Relation => argumentTypeLearner on p isNot "candidate" }
        }
    }
  }

  def legalArgumentsConstraint = sentences.ForEach { x: TextAnnotation =>
    // Each TextAnnotation is a sentence here.
    (sentences(x) ~> sentencesToRelations ~> relationsToPredicates).toList.distinct.ForAll { verb: Constituent =>
      val relationsList = (predicates(verb) ~> -relationsToPredicates).toList.distinct
      val legalArgumentsList = legalArguments(verb) :+ "candidate"

      // XXX - Verify the use of isOneOf here.
      relationsList.ForAll { rel: Relation => argumentTypeLearner on rel isOneOf legalArgumentsList }
    }
  }

  private val coreArguments = List("A0", "A1", "A2", "A3", "A4", "A5", "AA")
  def noDuplicatesConstraint = sentences.ForEach { x: TextAnnotation =>
    // Each TextAnnotation is a sentence here.
    (sentences(x) ~> sentencesToRelations ~> relationsToPredicates).toList.distinct.ForAll { verb: Constituent =>
      val relationsList = (predicates(verb) ~> -relationsToPredicates).toList.distinct
      coreArguments.ForAll { coreArgument: String =>
        relationsList.AtMost(1) { rel: Relation => argumentTypeLearner on rel is coreArgument }
      }
    }
  }

  /** constraint for reference to an actual argument/adjunct of type arg */
  private val referentialArguments = List("R-A1", "R-A2", "R-A3", "R-A4", "R-AA", "R-AM-ADV", "R-AM-CAU", "R-AM-EXT", "R-AM-LOC", "R-AM-MNR", "R-AM-PNC")
  def referenceConstraint = sentences.ForEach { x: TextAnnotation =>
    // Each TextAnnotation is a sentence here.
    (sentences(x) ~> sentencesToRelations ~> relationsToPredicates).toList.distinct.ForAll { verb: Constituent =>
      val relationsList = (predicates(verb) ~> -relationsToPredicates).toList.distinct
      referentialArguments.ForAll { rArg: String =>
        relationsList.ForAll { referentialRelation: Relation =>
          val otherArguments = relationsList.filter(_.getTarget != referentialRelation.getTarget)

          (argumentTypeLearner on referentialRelation is rArg) ==>
            otherArguments.Exists { rel: Relation => argumentTypeLearner on rel is rArg.substring(2) }
        }
      }
    }
  }

  /** constraint for continuity of an argument/adjunct of type arg */
  private val continuationArguments = List("C-A1", "C-A2", "C-A3", "C-A4", "C-A5", "C-AM-DIR", "C-AM-LOC", "C-AM-MNR", "C-AM-NEG", "C-AM-PNC")
  def continuationConstraint = sentences.ForEach { x: TextAnnotation =>
    // Each TextAnnotation is a sentence here.
    (sentences(x) ~> sentencesToRelations ~> relationsToPredicates).toList.distinct.ForAll { verb: Constituent =>
      val relationsList = (predicates(verb) ~> -relationsToPredicates).toList.distinct
      continuationArguments.ForAll { cArg: String =>
        relationsList.ForAll { continuationRelation: Relation =>
          val otherArguments = relationsList.filter(_.getTarget != continuationRelation.getTarget)

          (argumentTypeLearner on continuationRelation is cArg) ==>
            otherArguments.Exists { rel: Relation => argumentTypeLearner on rel is cArg.substring(2) }
        }
      }
    }
  }

  def filterConstraint = sentences.ForEach { x: TextAnnotation =>
    (sentences(x) ~> sentencesToRelations).toList.distinct.ForAll { rel: Relation =>
      (argumentXuIdentifierGivenApredicate on rel is "false") ==> (argumentTypeLearner on rel is "candidate")
    }
  }

  def arg_IdentifierClassifier_Constraint = relations.ForEach { x: Relation =>
    (argumentXuIdentifierGivenApredicate on x isFalse) ==> (argumentTypeLearner on x is "candidate")
  }

  //  def predArg_IdentifierClassifier_Constraint = relations.ForEach { x: Relation =>
  //    (predicateClassifier on x.getSource isTrue) and (argumentXuIdentifierGivenApredicate on x isTrue) ==>
  //      (argumentTypeLearner on x isNot "candidate")
  //  }
}

