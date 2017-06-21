/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling

import edu.illinois.cs.cogcomp.core.datastructures.ViewNames
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.{ PredicateArgumentView, TextAnnotation }
import edu.illinois.cs.cogcomp.core.utilities.DummyTextAnnotationGenerator
import edu.illinois.cs.cogcomp.nlp.corpusreaders.AbstractSRLAnnotationReader
import edu.illinois.cs.cogcomp.saulexamples.HighMemoryTest
import org.scalatest.{ FlatSpec, Matchers }

class AnnotatorTest extends FlatSpec with Matchers {
  val textAnnotation: TextAnnotation = DummyTextAnnotationGenerator.generateAnnotatedTextAnnotation(
    Array(ViewNames.POS, ViewNames.LEMMA, ViewNames.SHALLOW_PARSE, ViewNames.PARSE_GOLD),
    false,
    1
  )

  "SRLAnnotator" should "work" taggedAs (HighMemoryTest) in {
    val annotator = new SRLAnnotator(ViewNames.SRL_VERB)
    annotator.addView(textAnnotation)

    assert(textAnnotation.hasView(ViewNames.SRL_VERB), "SRL_VERB view should exist after annotation.")

    val srlView = textAnnotation.getView(ViewNames.SRL_VERB).asInstanceOf[PredicateArgumentView]
    assert(srlView.getPredicates.size() == 1)

    val verbPredicate = srlView.getPredicates.get(0)
    assert(srlView.getArguments(verbPredicate).size() >= 1)

    // Required attributes are populated.
    assert(verbPredicate.hasAttribute(AbstractSRLAnnotationReader.LemmaIdentifier))
    assert(verbPredicate.hasAttribute(AbstractSRLAnnotationReader.SenseIdentifier))
  }
}
