/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saulexamples.setcover

import edu.illinois.cs.cogcomp.lbjava.learn.Learner
import edu.illinois.cs.cogcomp.saul.classifier.infer.Constraint._
import edu.illinois.cs.cogcomp.saul.classifier.infer.ConstrainedClassifier
import edu.illinois.cs.cogcomp.saul.classifier.infer.solver.OJAlgo
import edu.illinois.cs.cogcomp.saul.datamodel.DataModel
import edu.illinois.cs.cogcomp.saul.lbjrelated.LBJLearnerEquivalent

object SetCoverSolverDataModel extends DataModel {

  val cities = node[City]

  val neighborhoods = node[Neighborhood]

  val cityContainsNeighborhoods = edge(cities, neighborhoods)

  cityContainsNeighborhoods.populateWith((c, n) => c == n.getParentCity)

  /** definition of the constraints */
  val containStation: LBJLearnerEquivalent = new LBJLearnerEquivalent {
    override val classifier: Learner = new ContainsStation()
  }

  def atLeastANeighborOfNeighborhoodIsCovered = { n: Neighborhood =>
    n.getNeighbors.Exists { neighbor: Neighborhood => containStation on neighbor isTrue }
  }

  def neighborhoodContainsStation = { n: Neighborhood =>
    (containStation on n) isTrue
  }

  def allCityNeighborhoodsAreCovered = { x: City =>
    x.getNeighborhoods.ForAll { n: Neighborhood =>
      neighborhoodContainsStation(n) or atLeastANeighborOfNeighborhoodIsCovered(n)
    }
  }

  def containsStationConstraint = SetCoverSolverDataModel.cities.ForAll { x: City => allCityNeighborhoodsAreCovered(x) }
}

object ConstrainedContainsStation extends ConstrainedClassifier[Neighborhood, City] {
  override lazy val onClassifier = SetCoverSolverDataModel.containStation
  override def pathToHead = Some(-SetCoverSolverDataModel.cityContainsNeighborhoods)
  override def subjectTo = Some(SetCoverSolverDataModel.containsStationConstraint)
  override def solverType = OJAlgo
}
