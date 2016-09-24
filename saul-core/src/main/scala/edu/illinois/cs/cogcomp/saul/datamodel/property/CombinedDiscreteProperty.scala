/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.datamodel.property

import java.util

import edu.illinois.cs.cogcomp.lbjava.classify.{ Classifier, FeatureVector }
import edu.illinois.cs.cogcomp.saul.datamodel.property.features.ClassifierContainsInLBP

import scala.reflect.ClassTag

case class CombinedDiscreteProperty[T <: AnyRef](atts: List[Property[T]])(implicit val tag: ClassTag[T]) extends TypedProperty[T, List[_]] {

  override val sensor: (T) => List[_] = {
    t: T => atts.map(att => att.sensor(t))
  }

  val name = "combined++" + atts.map(x => { x.name }).mkString("+")

  val packageName = "LBP_Package"

  val classifier = makeInternalClassifierWithName(name)

  override def featureVector(instance: T): FeatureVector = classifier.classify(instance)

  def makeInternalClassifierWithName(n: String): Classifier = new ClassifierContainsInLBP {
    this.containingPackage = packageName
    this.name = n

    override def getOutputType: String = "mixed%"

    def classify(instance: AnyRef): FeatureVector = {
      val t: T = instance.asInstanceOf[T]
      val featureVector = new FeatureVector()
      atts.foreach(_.addToFeatureVector(t, featureVector))
      featureVector
    }

    override def classify(examples: Array[AnyRef]): Array[FeatureVector] = {
      super.classify(examples)
    }

    override def getCompositeChildren: util.LinkedList[_] = {
      val result: util.LinkedList[Classifier] = new util.LinkedList[Classifier]()

      // TODO:bhargav - Check if we still need to do this.
      atts.foreach(x => result.add(Property.entitiesToLBJFeature(x)))
      result
    }

    override def discreteValue(example: AnyRef): String = {
      atts.head(example.asInstanceOf[T]).asInstanceOf[String]
    }
  }

  override def addToFeatureVector(instance: T, featureVector: FeatureVector): FeatureVector = {
    atts.foreach(_.addToFeatureVector(instance, featureVector))
    featureVector
  }
}
