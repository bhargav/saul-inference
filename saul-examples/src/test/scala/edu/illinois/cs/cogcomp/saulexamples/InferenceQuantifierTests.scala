/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saulexamples

import edu.illinois.cs.cogcomp.lbjava.learn.Learner
import edu.illinois.cs.cogcomp.saul.classifier.infer.Constraint._
import edu.illinois.cs.cogcomp.saul.classifier.infer.ConstrainedClassifier
import edu.illinois.cs.cogcomp.saul.classifier.infer.solver.OJAlgo
import edu.illinois.cs.cogcomp.saul.datamodel.DataModel
import edu.illinois.cs.cogcomp.saul.lbjrelated.LBJLearnerEquivalent
import edu.illinois.cs.cogcomp.saulexamples.setcover.{ City, ContainsStation, Neighborhood, SetCoverSolverDataModel }
import org.scalatest.{ FlatSpec, Matchers }

import scala.collection.JavaConversions._

class InferenceQuantifierTests extends FlatSpec with Matchers {

  object SomeDM extends DataModel {
    val cities = node[City]
    val neighborhoods = node[Neighborhood]
    val cityContainsNeighborhoods = edge(cities, neighborhoods)
    cityContainsNeighborhoods.populateWith((c, n) => c == n.getParentCity)

    val containStation = new ContainsStation()
    val containStationLBJEquivalent: LBJLearnerEquivalent = new LBJLearnerEquivalent {
      override val classifier: Learner = containStation
    }

    /** definition of the constraints */
    def neighborhoodContainsStation(n: Neighborhood) = containStationLBJEquivalent on n isTrue

    def atLeastSomeNeighborsAreCoveredConstraint = cities.ForAll { x: City =>
      x.getNeighborhoods.AtLeast(2) { n: Neighborhood => neighborhoodContainsStation(n) }
    }

    def atLeastSomeNeighborsAreCoveredConstraintUsingAtMost = cities.ForAll { x: City =>
      !x.getNeighborhoods.AtMost(2) { n: Neighborhood => neighborhoodContainsStation(n) }
    }

    def allNeighborsAreCoveredConstraint = cities.ForAll { x: City =>
      x.getNeighborhoods.ForAll { n: Neighborhood => neighborhoodContainsStation(n) }
    }

    def singleNeighborsAreCoveredConstraint = cities.ForAll { x: City =>
      x.getNeighborhoods.Exists { n: Neighborhood => neighborhoodContainsStation(n) }
    }
  }

  import SomeDM._
  object AtLeastSomeNeighborhoods extends ConstrainedClassifier[Neighborhood, City] {
    override val pathToHead = Some(-cityContainsNeighborhoods)
    override def subjectTo = Some(atLeastSomeNeighborsAreCoveredConstraint)
    override val solverType = OJAlgo
    override def onClassifier: LBJLearnerEquivalent = containStationLBJEquivalent
  }

  object AtLeastSomeNeighborhoodsUsingAtMost extends ConstrainedClassifier[Neighborhood, City] {
    override val pathToHead = Some(-cityContainsNeighborhoods)
    override def subjectTo = Some(atLeastSomeNeighborsAreCoveredConstraintUsingAtMost)
    override val solverType = OJAlgo
    override def onClassifier: LBJLearnerEquivalent = containStationLBJEquivalent
  }

  object AllNeighborhoods extends ConstrainedClassifier[Neighborhood, City] {
    override val pathToHead = Some(-cityContainsNeighborhoods)
    override def subjectTo = Some(allNeighborsAreCoveredConstraint)
    override val solverType = OJAlgo
    override def onClassifier: LBJLearnerEquivalent = containStationLBJEquivalent
  }

  object ASingleNeighborhood extends ConstrainedClassifier[Neighborhood, City] {
    override val pathToHead = Some(-cityContainsNeighborhoods)
    override def subjectTo = Some(singleNeighborsAreCoveredConstraint)
    override val solverType = OJAlgo
    override def onClassifier: LBJLearnerEquivalent = containStationLBJEquivalent
  }

  val cityInstances = new City("../saul-examples/src/test/resources/SetCover/example.txt")
  val neighborhoodInstances = cityInstances.getNeighborhoods.toList

  SomeDM.cities populate List(cityInstances)
  SomeDM.neighborhoods populate neighborhoodInstances
  def getParentCity = (n: Neighborhood) => n.getParentCity
  SomeDM.cityContainsNeighborhoods.populateWith((c: City, n: Neighborhood) => n.getParentCity == c)

  "Quantifier atleast " should " work " in {
    cityInstances.getNeighborhoods.count(n => AtLeastSomeNeighborhoods(n) == "true") should be(2)
  }

  // negation of atmost(2) is equivalent to atleast(2)
  "Quantifier atmost " should " work " in {
    cityInstances.getNeighborhoods.count(n => AtLeastSomeNeighborhoodsUsingAtMost(n) == "true") should be(2)
    info("cityInstances.getNeighborhoods: " + cityInstances.getNeighborhoods.size())
  }

  "Quantifier forall " should " work " in {
    cityInstances.getNeighborhoods.count(n => AllNeighborhoods(n) == "true") should be(9)
  }

  "Quantifier exists " should " work " in {
    cityInstances.getNeighborhoods.count(n => ASingleNeighborhood(n) == "true") should be(1)
  }
}
