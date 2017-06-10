/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saulexamples.nlp.EntityRelation

import edu.illinois.cs.cogcomp.saul.classifier.infer.solver._
import edu.illinois.cs.cogcomp.saul.classifier.infer.ConstrainedClassifier
import edu.illinois.cs.cogcomp.saulexamples.EntityMentionRelation.datastruct.{ ConllRawToken, ConllRelation }

object EntityRelationConstrainedClassifiers {
  object OrgConstrainedClassifier extends ConstrainedClassifier[ConllRawToken, ConllRelation] {
    override lazy val onClassifier = EntityRelationClassifiers.OrganizationClassifier
    override def pathToHead = Some(EntityRelationDataModel.tokenToPair)
    override def subjectTo = Some(EntityRelationConstraints.relationArgumentConstraints)
    override def solverType = OJAlgo
  }

  object PerConstrainedClassifier extends ConstrainedClassifier[ConllRawToken, ConllRelation] {
    override lazy val onClassifier = EntityRelationClassifiers.PersonClassifier
    override def pathToHead = Some(EntityRelationDataModel.tokenToPair)
    override def subjectTo = Some(EntityRelationConstraints.relationArgumentConstraints)
    override def solverType = OJAlgo
  }

  object LocConstrainedClassifier extends ConstrainedClassifier[ConllRawToken, ConllRelation] {
    override lazy val onClassifier = EntityRelationClassifiers.LocationClassifier
    override def pathToHead = Some(EntityRelationDataModel.tokenToPair)
    override def subjectTo = Some(EntityRelationConstraints.relationArgumentConstraints)
    override def solverType = OJAlgo
  }

  object WorksForRelationConstrainedClassifier extends ConstrainedClassifier[ConllRelation, ConllRelation] {
    override lazy val onClassifier = EntityRelationClassifiers.WorksForClassifier
    override def subjectTo = Some(EntityRelationConstraints.relationArgumentConstraints)
    override def solverType = OJAlgo
  }

  object LivesInRelationConstrainedClassifier extends ConstrainedClassifier[ConllRelation, ConllRelation] {
    override lazy val onClassifier = EntityRelationClassifiers.LivesInClassifier
    override def subjectTo = Some(EntityRelationConstraints.relationArgumentConstraints)
    override def solverType = OJAlgo
  }
}

