/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.factorgraph

import cc.factorie.Factorie.BooleanVariable

class BinaryRandomVariable(val initialValue: Boolean, classifier: String = "") extends BooleanVariable(initialValue) {
  override def toString(): String = if (classifier.isEmpty) super.toString() else classifier
}
