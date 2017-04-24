/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.solver

import edu.illinois.cs.cogcomp.infer.ilp.ILPSolver
import edu.illinois.cs.cogcomp.saul.classifier.infer._
import edu.illinois.cs.cogcomp.saul.lbjrelated.LBJLearnerEquivalent

import scala.collection._

private[solver] class ILPInferenceManager {
  // a small number used in creation of exclusive inequalities
  private val epsilon = 0.01

  // for each estimator, maps the label of the estimator, to the integer label of the solver
  val estimatorToSolverLabelMap = mutable.Map[LBJLearnerEquivalent, mutable.Map[_, Seq[(Int, String)]]]()

  // greater or equal to: ax >= b
  case class ILPInequalityGEQ(a: Array[Double], x: Array[Int], b: Double)

  def processConstraints[V <: Any](saulConstraint: Constraint[V], solver: ILPSolver): Set[ILPInequalityGEQ] = {

    saulConstraint match {
      case c: PropositionalEqualityConstraint[V] =>
        assert(c.instanceOpt.isDefined, "the instance in the constraint should definitely be defined.")

        // add the missing variables to the map
        addVariablesToInferenceProblem(Seq(c.instanceOpt.get), c.estimator, solver)

        // estimates per instance
        val estimatorScoresMap = estimatorToSolverLabelMap(c.estimator).asInstanceOf[mutable.Map[V, Seq[(Int, String)]]]

        val (ilpIndices, labels) = estimatorScoresMap.get(c.instanceOpt.get) match {
          case Some(ilpIndexLabelPairs) => ilpIndexLabelPairs.unzip
          case None =>
            val confidenceScores = c.estimator.classifier.scores(c).toArray.map(_.score)
            val labels = c.estimator.classifier.scores(c.instanceOpt.get).toArray.map(_.value)
            val indicesPerLabels = solver.addDiscreteVariable(confidenceScores)
            estimatorScoresMap += (c.instanceOpt.get -> indicesPerLabels.zip(labels))
            estimatorToSolverLabelMap.put(c.estimator, estimatorScoresMap)
            (indicesPerLabels.toSeq, labels.toSeq)
        }

        assert(
          c.inequalityValOpt.isEmpty || c.equalityValOpt.isEmpty,
          s"the equality constraint $c is not completely defined"
        )
        assert(
          c.inequalityValOpt.isDefined || c.equalityValOpt.isDefined,
          s"the equality constraint $c has values for both equality and inequality"
        )

        val classifierTagSet = if (c.estimator.classifier.getLabeler != null) {
          (0 until c.estimator.classifier.getLabelLexicon.size).map { i =>
            c.estimator.classifier.getLabelLexicon.lookupKey(i).getStringValue
          }.toSet
        } else {
          c.estimator.classifier.allowableValues().toSet
        }

        if (c.equalityValOpt.isDefined) {
          // first make sure the target value is valid
          require(
            classifierTagSet.contains(c.equalityValOpt.get),
            s"The target value ${c.equalityValOpt} is not a valid value for classifier ${c.estimator} - " +
              s"the classifier tag-set is $classifierTagSet"
          )
          val labelIndexOpt = labels.zipWithIndex.collectFirst { case (label, idx) if label == c.equalityValOpt.get => idx }
          val x = ilpIndices(labelIndexOpt.getOrElse(
            throw new Exception(s"the corresponding index to label ${c.equalityValOpt.get} not found")
          ))

          // 1x == 1; which can be written as
          // (a) +1x >= +1
          // (b) -1x >= -1
          // ILPInequalityGEQ(Array(-1.0), Array(x), -1.0)
          Set(ILPInequalityGEQ(Array(1.0), Array(x), 1.0)) //, ILPInequalityGEQ(Array(-1.0), Array(x), -1.0))
        } else {
          require(
            classifierTagSet.contains(c.inequalityValOpt.get),
            s"The target value ${c.inequalityValOpt} is not a valid value for classifier ${c.estimator} " +
              s"with the tag-set: $classifierTagSet"
          )
          val labelIndexOpt = labels.zipWithIndex.collectFirst { case (label, idx) if label == c.inequalityValOpt.get => idx }
          val x = ilpIndices(labelIndexOpt.getOrElse(
            throw new Exception(s"the corresponding index to label ${c.equalityValOpt.get} not found")
          ))
          // 1 x == 0 : possible only when x = 0, which can be written as
          // (a) +1 x >= 0
          // (b) -1 x >= 0
          Set(ILPInequalityGEQ(Array(-1.0), Array(x), 0.0)) //, ILPInequalityGEQ(Array(1.0), Array(x), 0.0))
        }
      case c: EstimatorPairEqualityConstraint[V] =>
        assert(c.estimator2Opt.isDefined, "the second estimator is not defined for estimator-pair constraint. Weird . . . ")

        // add the missing variables to the map
        addVariablesToInferenceProblem(Seq(c.instance), c.estimator1, solver)
        addVariablesToInferenceProblem(Seq(c.instance), c.estimator2Opt.get, solver)

        // estimates per instance
        val estimatorScoresMap1 = estimatorToSolverLabelMap(c.estimator1).asInstanceOf[mutable.Map[V, Seq[(Int, String)]]]
        val estimatorScoresMap2 = estimatorToSolverLabelMap(c.estimator2Opt.get).asInstanceOf[mutable.Map[V, Seq[(Int, String)]]]

        val labelToIndices1 = estimatorScoresMap1.get(c.instance) match {
          case Some(ilpIndexLabelPairs) => ilpIndexLabelPairs.map(_.swap).toMap
          case None => throw new Exception("The instance hasn't been seen??")
        }

        val labelToIndices2 = estimatorScoresMap2.get(c.instance) match {
          case Some(ilpIndexLabelPairs) => ilpIndexLabelPairs.map(_.swap).toMap
          case None => throw new Exception("The instance hasn't been seen??")
        }

        // this requirement might be an overkill, but keeping it for now.
        require(
          labelToIndices1.keySet == labelToIndices2.keySet,
          "the label set for the two classifiers is not the same"
        )

        val yAll = solver.addDiscreteVariable(Array.fill(labelToIndices1.keySet.size) { 0 })

        val labels = labelToIndices1.keySet.toSeq
        labels.indices.flatMap { idx =>
          val label = labels(idx)
          val y = yAll(idx)
          val variable1 = labelToIndices1(label)
          val variable2 = labelToIndices2(label)

          if (c.equalsOpt.get) {
            // for each variable, if y is active, that variable should also be active:
            // 1 variable >= 1 y
            Set(
              ILPInequalityGEQ(Array(1.0, -1.0), Array(variable1, y), 0.0),
              ILPInequalityGEQ(Array(1.0, -1.0), Array(variable2, y), 0.0)
            )
          } else {
            // for each variable, if y is active, that variable should also be active:
            // variable >= y  which is, variable - y >= 0
            // 1 - variable >= 1 y which is, -variable -y >= -1
            Set(
              ILPInequalityGEQ(Array(1.0, -1.0), Array(variable1, y), 0.0),
              ILPInequalityGEQ(Array(-1.0, -1.0), Array(variable1, y), -1.0),
              ILPInequalityGEQ(Array(1.0, -1.0), Array(variable2, y), 0.0),
              ILPInequalityGEQ(Array(-1.0, -1.0), Array(variable2, y), -1.0)
            )
          }
        }.toSet

      case c: InstancePairEqualityConstraint[V] =>
        assert(c.instance2Opt.isDefined, "the second instance is not defined for estimator-pair constraint. Weird . . . ")

        // add the missing variables to the map
        addVariablesToInferenceProblem(Seq(c.instance1), c.estimator, solver)
        addVariablesToInferenceProblem(Seq(c.instance2Opt.get), c.estimator, solver)

        // estimates per instance
        val estimatorScoresMap = estimatorToSolverLabelMap(c.estimator).asInstanceOf[mutable.Map[V, Seq[(Int, String)]]]

        val labelToIndices1 = estimatorScoresMap.get(c.instance1) match {
          case Some(ilpIndexLabelPairs) => ilpIndexLabelPairs.map(_.swap).toMap
          case None => throw new Exception("The instance hasn't been seen??")
        }

        val labelToIndices2 = estimatorScoresMap.get(c.instance2Opt.get) match {
          case Some(ilpIndexLabelPairs) => ilpIndexLabelPairs.map(_.swap).toMap
          case None => throw new Exception("The instance hasn't been seen??")
        }

        // this requirement might be an overkill, but keeping it for now.
        require(
          labelToIndices1.keySet == labelToIndices2.keySet,
          "the label set for the two classifiers is not the same; " +
            "although they belong to the same classifier; weird . . . "
        )

        val yAll = solver.addDiscreteVariable(Array.fill(labelToIndices1.keySet.size) { 0 })

        val labels = labelToIndices1.keySet.toSeq
        labels.indices.flatMap { idx =>
          val label = labels(idx)
          val y = yAll(idx)
          val variable1 = labelToIndices1(label)
          val variable2 = labelToIndices2(label)

          if (c.equalsOpt.get) {
            // for each variable, if y is active, that variable should also be active:
            // 1 variable >= 1 y
            Set(
              ILPInequalityGEQ(Array(1.0, -1.0), Array(variable1, y), 0.0),
              ILPInequalityGEQ(Array(1.0, -1.0), Array(variable2, y), 0.0)
            )
          } else {
            // for each variable, if y is active, that variable should also be active:
            // variable >= y  which is, variable - y >= 0
            // 1 - variable >= 1 y which is, -variable -y >= -1
            Set(
              ILPInequalityGEQ(Array(1.0, -1.0), Array(variable1, y), 0.0),
              ILPInequalityGEQ(Array(-1.0, -1.0), Array(variable1, y), -1.0),
              ILPInequalityGEQ(Array(1.0, -1.0), Array(variable2, y), 0.0),
              ILPInequalityGEQ(Array(-1.0, -1.0), Array(variable2, y), -1.0)
            )
          }
        }.toSet

      case c: PairConjunction[V, _] =>
        val InequalitySystem1 = processConstraints(c.c1, solver)
        val InequalitySystem2 = processConstraints(c.c2, solver)

        // conjunction is simple; you just include all the inequalities
        InequalitySystem1 union InequalitySystem2
      case c: PairDisjunction[V, _] =>
        val InequalitySystem1 = processConstraints(c.c1, solver)
        val InequalitySystem2 = processConstraints(c.c2, solver)
        val y1 = solver.addBooleanVariable(0.0)
        val y2 = solver.addBooleanVariable(0.0)
        // a1.x >= b1 or a2.x >= b2:
        // should be converted to
        // a1.x >= b1.y1 + min(a1.x).(1-y1)
        // a2.x >= b2.(1-y2) + min(a2.x).y2
        // y1 + y2 >= 1
        // We can summarize the first one as:
        // newA1 = [a1, min(a1.x)-b1]
        // newX1 = [x, y]
        // newB1 = min(a1.x)
        val InequalitySystem1New = InequalitySystem1.map { ins =>
          val minValue = (ins.a.filter(_ < 0) :+ 0.0).sum
          val a1New = ins.a :+ (minValue - ins.b)
          val x1New = ins.x :+ y1
          val b1New = minValue
          ILPInequalityGEQ(a1New, x1New, b1New)
        }
        val InequalitySystem2New = InequalitySystem2.map { ins =>
          val minValue = (ins.a.filter(_ < 0) :+ 0.0).sum
          val a2New = ins.a :+ (minValue - ins.b)
          val x2New = ins.x :+ y2
          val b2New = minValue
          ILPInequalityGEQ(a2New, x2New, b2New)
        }
        val atLeastOne = ILPInequalityGEQ(Array(1, 1), Array(y1, y2), 1.0)
        InequalitySystem1New union InequalitySystem2New + atLeastOne
      case c: Negation[V] =>
        // change the signs of the coefficients
        val InequalitySystemToBeNegated = processConstraints(c.p, solver)
        InequalitySystemToBeNegated.map { in =>
          val minusA = in.a.map(-_)
          val minusB = -in.b + epsilon
          ILPInequalityGEQ(minusA, in.x, minusB)
        }
      case c: AtLeast[V, _] =>
        val InequalitySystems = c.constraints.map { processConstraints(_, solver) }
        // for each inequality ax >= b we introduce a binary variable y
        // and convert the constraint to ax >= by + (1-y).min(ax) and ax < (b-e)(1-y) + y.max(ax)
        // which can be written as ax+y(min(ax)-b) >= min(ax) and ax + y(b - e - max(ax)) < b - e
        // newA1 = [a, min(ax) - b]
        // newX1 = [x, y]
        // newB1 = min(ax)
        // newA2 = [-a,  max(ax) - b]
        // newX2 = [x, y]
        // newB2 = -b + epsilon
        val (inequalities, newAuxillaryVariables) = InequalitySystems.map { inequalitySystem =>
          val y = solver.addBooleanVariable(0.0)
          val newInequalities = inequalitySystem.flatMap { inequality =>
            val maxValue = (inequality.a.filter(_ > 0) :+ 0.0).sum
            val minValue = (inequality.a.filter(_ < 0) :+ 0.0).sum
            val newA1 = inequality.a :+ (minValue - inequality.b)
            val newX1 = inequality.x :+ y
            val newB1 = minValue
            val newA2 = inequality.a.map(-_) :+ (maxValue + epsilon - inequality.b)
            val newX2 = inequality.x :+ y
            val newB2 = -inequality.b + epsilon
            Set(ILPInequalityGEQ(newA1, newX1, newB1), ILPInequalityGEQ(newA2, newX2, newB2))
          }
          (newInequalities, y)
        }.unzip
        // add a new constraint: at least k constraints should be active
        inequalities.flatten +
          ILPInequalityGEQ(newAuxillaryVariables.toArray.map(_ => 1.0), newAuxillaryVariables.toArray, c.k)
      case c: AtMost[V, _] =>
        val InequalitySystems = c.constraints.map { processConstraints(_, solver) }
        // for each inequality ax >= b we introduce a binary variable y
        // and convert the constraint to ax >= by + (1-y).min(ax) and ax < (b-e)(1-y) + y.max(ax)
        // which can be written as ax+y(min(ax)-b) >= min(ax) and ax + y(b - e - max(ax)) < b - e
        // newA1 = [a, min(ax) - b]
        // newX1 = [x, y]
        // newB1 = min(ax)
        // newA2 = [-a,  max(ax) - b]
        // newX2 = [x, y]
        // newB2 = -b + epsilon
        val (inequalities, newAuxillaryVariables) = InequalitySystems.map { inequalitySystem =>
          val y = solver.addBooleanVariable(0.0)
          val newInequalities = inequalitySystem.flatMap { inequality =>
            val maxValue = (inequality.a.filter(_ > 0) :+ 0.0).sum
            val minValue = (inequality.a.filter(_ < 0) :+ 0.0).sum
            val newA1 = inequality.a :+ (minValue - inequality.b)
            val newX1 = inequality.x :+ y
            val newB1 = minValue
            val newA2 = inequality.a.map(-_) :+ (maxValue + epsilon - inequality.b)
            val newX2 = inequality.x :+ y
            val newB2 = -inequality.b + epsilon
            Set(ILPInequalityGEQ(newA1, newX1, newB1), ILPInequalityGEQ(newA2, newX2, newB2))
          }
          (newInequalities, y)
        }.unzip
        // add a new constraint: at least k constraints should be active
        inequalities.flatten +
          ILPInequalityGEQ(newAuxillaryVariables.toArray.map(_ => -1.0), newAuxillaryVariables.toArray, -c.k)
      case c: Exactly[V, _] =>
        val InequalitySystems = c.constraints.map { processConstraints(_, solver) }
        // for each inequality ax >= b we introduce a binary variable y
        // and convert the constraint to ax >= by + (1-y).min(ax) and ax < (b-e)(1-y) + y.max(ax)
        // which can be written as ax+y(min(ax)-b) >= min(ax) and ax + y(b - e - max(ax)) < b - e
        // newA1 = [a, min(ax) - b]
        // newX1 = [x, y]
        // newB1 = min(ax)
        // newA2 = [-a,  max(ax) - b]
        // newX2 = [x, y]
        // newB2 = -b + epsilon
        val (inequalities, newAuxillaryVariables) = InequalitySystems.map { inequalitySystem =>
          val y = solver.addBooleanVariable(0.0)
          val newInequalities = inequalitySystem.flatMap { inequality =>
            val maxValue = (inequality.a.filter(_ > 0) :+ 0.0).sum
            val minValue = (inequality.a.filter(_ < 0) :+ 0.0).sum
            val newA1 = inequality.a :+ (minValue - inequality.b)
            val newX1 = inequality.x :+ y
            val newB1 = minValue
            val newA2 = inequality.a.map(-_) :+ (maxValue + epsilon - inequality.b)
            val newX2 = inequality.x :+ y
            val newB2 = -inequality.b + epsilon
            Set(ILPInequalityGEQ(newA1, newX1, newB1), ILPInequalityGEQ(newA2, newX2, newB2))
          }
          (newInequalities, y)
        }.unzip
        // add a new constraint: at least k constraints should be active
        inequalities.flatten union Set(
          ILPInequalityGEQ(newAuxillaryVariables.toArray.map(_ => 1.0), newAuxillaryVariables.toArray, c.k),
          ILPInequalityGEQ(newAuxillaryVariables.toArray.map(_ => -1.0), newAuxillaryVariables.toArray, -c.k)
        )
      case c: ForAll[V, _] =>
        c.constraints.flatMap { processConstraints(_, solver) }
      case _ =>
        throw new Exception("Saul implication is converted to other operations. ")
    }
  }

  // if the estimator has never been seen before, add its labels to the map
  def createEstimatorSpecificCache[V](estimator: LBJLearnerEquivalent): Unit = {
    if (!estimatorToSolverLabelMap.keySet.contains(estimator)) {
      estimatorToSolverLabelMap += (estimator -> mutable.Map[V, Seq[(Int, String)]]())
    }
  }

  def addVariablesToInferenceProblem[V](instances: Seq[V], estimator: LBJLearnerEquivalent, solver: ILPSolver): Unit = {
    createEstimatorSpecificCache(estimator)

    // estimates per instance
    val estimatorScoresMap = estimatorToSolverLabelMap(estimator).asInstanceOf[mutable.Map[V, Seq[(Int, String)]]]

    // adding the estimates to the solver and to the map
    instances.foreach { c =>
      val confidenceScores = estimator.classifier.scores(c).toArray.map(_.score)
      val labels = estimator.classifier.scores(c).toArray.map(_.value)
      val instanceIndexPerLabel = solver.addDiscreteVariable(confidenceScores)
      if (!estimatorScoresMap.contains(c)) {
        estimatorScoresMap += (c -> instanceIndexPerLabel.zip(labels).toSeq)
      }
    }

    // add the variables back into the map
    estimatorToSolverLabelMap.put(estimator, estimatorScoresMap)
  }
}

object ILPInferenceManager {
  /** Contains cache of problems already solved. The key is the head object, which maps to instances and their
    * predicted values in the output of inference
    */
  val cachedResults = mutable.Map[String, (ILPSolver, LBJLearnerEquivalent, mutable.Map[LBJLearnerEquivalent, mutable.Map[_, Seq[(Int, String)]]])]()
}
