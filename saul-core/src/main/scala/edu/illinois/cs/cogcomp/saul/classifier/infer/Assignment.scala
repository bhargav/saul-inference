/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer

import edu.illinois.cs.cogcomp.lbjava.classify.ScoreSet
import edu.illinois.cs.cogcomp.saul.lbjrelated.LBJLearnerEquivalent

import scala.collection.mutable

// (bhargav) - Can we make this immutable?
case class Assignment(learner: LBJLearnerEquivalent) extends mutable.HashMap[Any, ScoreSet]