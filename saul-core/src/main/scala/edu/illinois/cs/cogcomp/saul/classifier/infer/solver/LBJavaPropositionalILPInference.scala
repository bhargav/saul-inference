/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.solver

import java.util
import java.util.Date

import edu.illinois.cs.cogcomp.infer.ilp.ILPSolver
import edu.illinois.cs.cogcomp.lbjava.classify.{ Score, ScoreSet }
import edu.illinois.cs.cogcomp.lbjava.infer._
import edu.illinois.cs.cogcomp.lbjava.learn.Learner
import edu.illinois.cs.cogcomp.saul.util.Logging

import scala.collection.JavaConverters._

// Only supports Propositional Constraints
class LBJavaPropositionalILPInference(solver: ILPSolver, verbosity: Int = ILPInference.VERBOSITY_NONE) extends ILPInference(solver, verbosity) with PropositionalInference with Logging {
  // Hide the method that accepts constraints in logical form.
  override def addConstraint(c: FirstOrderConstraint): Unit = ???

  /** Add a Propositional Constraint to the system of constraints to solve.
    * @param c Propositional constraint to consider.
    */
  override def addConstraint(c: PropositionalConstraint, variablesToConsider: Seq[FirstOrderVariable]): Unit = {
    solver.reset()

    if (constraint == null)
      constraint = c
    else
      constraint = new PropositionalConjunction(constraint.asInstanceOf[PropositionalConstraint], c)

    // Calling getVariable adds the variable to the `variables` field that is used in the `infer` method.
    variablesToConsider.foreach(getVariable)
  }

  override def infer(): Unit = {
    if (tautology || solver.isSolved) {
      logger.debug("Already solved.")
      return
    }

    val indexMapLocal = new util.HashMap[PropositionalVariable, Integer]()

    solver.setMaximize(true)
    constraint.consolidateVariables(variables)

    if (verbosity > ILPInference.VERBOSITY_NONE) logger.info("variables: (" + new Date() + ")")

    val distinctVariablesWithScores = variables.values
      .asScala
      .groupBy({
        case fov: FirstOrderVariable =>
          (fov.getClassifier, fov.getExample)
        case pv: PropositionalVariable =>
          (pv.getClassifier, pv.getExample)
      })
      .map(_._2.head)
      .map({
        case fov: FirstOrderVariable =>
          (fov.getClassifier, fov.getExample, fov.getScores)
        case pv: PropositionalVariable =>
          (pv.getClassifier, pv.getExample, pv.getClassifier.scores(pv.getExample))
      })

    distinctVariablesWithScores
      .foreach({
        case (classifier: Learner, instance: Any, scoreset: ScoreSet) =>
          val ss = getNormalizer(classifier).normalize(scoreset)
          val scores = ss.toArray

          // putting scores in a real-valued array
          val weights = scores.map(_.score)
          val indexes = solver.addDiscreteVariable(weights)

          scores.zipWithIndex.foreach({
            case (score: Score, idx: Int) =>
              indexMapLocal.put(new PropositionalVariable(classifier, instance, score.value), new Integer(indexes(idx)))

              if (verbosity >= ILPInference.VERBOSITY_HIGH) {
                val toPrint = new StringBuffer()
                toPrint.append("x_")
                toPrint.append(indexes(idx))

                while (toPrint.length() < 8)
                  toPrint.insert(0, ' ')

                toPrint.append(" (")
                toPrint.append(score.score)
                toPrint.append("): ")
                toPrint.append(classifier)
                toPrint.append("(")
                toPrint.append(Inference.exampleToString(instance))
                toPrint.append(") == ")
                toPrint.append(score.value)

                logger.info(toPrint.toString)
              }
          })
      })

    indexMap = indexMapLocal

    if (verbosity > ILPInference.VERBOSITY_NONE) logger.info("simplification: (" + new Date() + ")")

    val propositional = {
      constraint match {
        case propositionalConjunction: PropositionalConjunction => propositionalConjunction.simplify(true)
        case _ => constraint.asInstanceOf[PropositionalConstraint].simplify()
      }
    }

    if (propositional.isInstanceOf[PropositionalConstant]) {
      if (propositional.evaluate()) {
        tautology = true
        return
      } else {
        logger.error("ILP ERROR: Unsatisfiable constraints!")
        solver.addEqualityConstraint(Array(0), Array(1.0), 2)
      }
    }

    if (verbosity > ILPInference.VERBOSITY_NONE) logger.info("translation: (" + new Date() + ")")

    topLevel = true
    propositional.runVisit(this)

    if (verbosity > ILPInference.VERBOSITY_NONE) logger.info("solution: (" + new Date() + ")")

    if (!solver.solve())
      throw new InferenceNotOptimalException(solver, head)

    if (verbosity > ILPInference.VERBOSITY_NONE) logger.info("variables set true in solution: (" + new Date() + ")")

    var variableIndex: Int = 0
    variables.values()
      .asScala
      .filter(_.isInstanceOf[FirstOrderVariable])
      .foreach({ variable: Any =>
        val v = variable.asInstanceOf[FirstOrderVariable]
        val scores = v.getClassifier.scores(v.getExample).toArray

        scores.foreach({ score: Score =>
          if (solver.getBooleanValue(variableIndex)) {
            v.setValue(score.value)

            if (verbosity >= ILPInference.VERBOSITY_HIGH) {
              val toPrint = new StringBuffer()
              toPrint.append("x_")
              toPrint.append(variableIndex)

              while (toPrint.length() < 8)
                toPrint.insert(0, ' ')

              toPrint.append(": ")
              toPrint.append(v)

              logger.info(toPrint.toString)
            }
          }

          variableIndex += 1
        })
      })
  }
}
