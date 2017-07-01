/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling

import edu.illinois.cs.cogcomp.core.datastructures.textannotation.Relation
import edu.illinois.cs.cogcomp.lbjava.classify.{ Score, ScoreSet }

import scala.collection.mutable

// Note: Targeted to in PredicateArgumentViews only.
// Used in SRL Annotation.
object GreedyDecoder {
  def decodeNoOverlap(inputRelationWithScores: Seq[(Relation, ScoreSet)], labelsToExclude: Set[String] = Set.empty): Seq[(Relation, Score)] = {
    val filterExcludes = inputRelationWithScores.map({
      case (relation, scoreset) =>
        val highScoreLabel = scoreset.highScoreValue()
        (relation, scoreset.getScore(highScoreLabel))
    }).filterNot(x => labelsToExclude.contains(x._2.value))

    if (filterExcludes.isEmpty) {
      return Seq.empty
    }

    val minSpan = filterExcludes.map(_._1.getTarget.getStartSpan).min
    val maxSpan = filterExcludes.map(_._1.getTarget.getEndSpan).max

    val range = maxSpan - minSpan + 1
    val spanPosition = new mutable.BitSet(range)

    // Check if the sorting is stable
    filterExcludes.sortBy(x => -x._2.score) // Sort from largest to lowest scores
      .flatMap({
        case (relation, score) =>
          val startSpan = relation.getTarget.getStartSpan
          val endSpan = relation.getTarget.getEndSpan - 1
          val hasOverlap = (startSpan to endSpan).exists(sp => spanPosition.contains(sp - minSpan))

          if (!hasOverlap) {
            spanPosition ++= Range(startSpan - minSpan, endSpan - minSpan + 1)
            Some((relation, score))
          } else {
            None
          }
      })
  }
}
