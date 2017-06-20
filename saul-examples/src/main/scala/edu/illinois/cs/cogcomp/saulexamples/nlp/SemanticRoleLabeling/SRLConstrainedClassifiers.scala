/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling

import edu.illinois.cs.cogcomp.core.datastructures.textannotation.{ Relation, TextAnnotation }
import edu.illinois.cs.cogcomp.saul.classifier.infer.ConstrainedClassifier
import edu.illinois.cs.cogcomp.saul.classifier.infer.solver.Gurobi
import edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling.SRLClassifiers.{ argumentTypeLearner, argumentXuIdentifierGivenApredicate }
import edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling.SRLConstraints._

object SRLConstrainedClassifiers {
  import SRLApps.srlDataModelObject._

  object ArgTypeConstrainedClassifier extends ConstrainedClassifier[Relation, TextAnnotation] {
    override def subjectTo = Some(argumentTypeConstraints)
    override def solverType = Gurobi
    override lazy val onClassifier = argumentTypeLearner
    override val pathToHead = Some(-sentencesToRelations)
  }

  object ArgIsTypeConstrainedClassifier extends ConstrainedClassifier[Relation, Relation] {
    override def subjectTo = Some(arg_IdentifierClassifier_Constraint)
    override def solverType = Gurobi
    override lazy val onClassifier = argumentTypeLearner
  }

  object ArgIdentifyConstrainedClassifier extends ConstrainedClassifier[Relation, Relation] {
    override def subjectTo = Some(arg_IdentifierClassifier_Constraint)
    override def solverType = Gurobi
    override lazy val onClassifier = argumentXuIdentifierGivenApredicate
  }
}
