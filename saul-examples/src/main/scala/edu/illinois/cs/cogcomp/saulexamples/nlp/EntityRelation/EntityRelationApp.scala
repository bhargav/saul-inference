/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saulexamples.nlp.EntityRelation

import edu.illinois.cs.cogcomp.core.datastructures.ViewNames
import edu.illinois.cs.cogcomp.nlp.tokenizer.StatefulTokenizer
import edu.illinois.cs.cogcomp.nlp.utility.TokenizerTextAnnotationBuilder
import edu.illinois.cs.cogcomp.saul.classifier.{ ClassifierUtils, JointTrainSparseNetwork, Learnable }
import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.illinois.cs.cogcomp.saulexamples.EntityMentionRelation.datastruct.{ ConllRawSentence, ConllRelation }
import edu.illinois.cs.cogcomp.saulexamples.nlp.EntityRelation.EntityRelationClassifiers._
import edu.illinois.cs.cogcomp.saulexamples.nlp.EntityRelation.EntityRelationConstrainedClassifiers._
import edu.illinois.cs.cogcomp.saulexamples.nlp.EntityRelation.EntityRelationDataModel._
import edu.illinois.cs.cogcomp.saulexamples.nlp.POSTagger.POSTaggerApp

import scala.io.StdIn
import scala.util.Random

object EntityRelationApp extends Logging {
  // learned models from the "saul-er-models" jar package
  val jarModelPath = "edu/illinois/cs/cogcomp/saulexamples/nlp/EntityRelation/models/"

  def main(args: Array[String]): Unit = {
    /** Choose the experiment you're interested in by changing the following line */
    val testType = ERExperimentType.FactorConstrained

    testType match {
      case ERExperimentType.IndependentClassifiers => trainIndependentClassifiers()
      case ERExperimentType.IndependentClassifiersCV => trainIndependentClassifiersCV()
      case ERExperimentType.TestFromModel => testIndependentClassifiers()
      case ERExperimentType.TestFromModelCV => testIndependentClassifiersCV()
      case ERExperimentType.PipelineTraining => runPipelineTraining()
      case ERExperimentType.PipelineTestFromModel => testPipelineRelationModels()
      case ERExperimentType.LPlusI => runLPlusI()
      case ERExperimentType.LPlusICV => runLPlusICV()
      case ERExperimentType.JointTraining => runJointTraining()
      case ERExperimentType.InteractiveMode => interactiveWithPretrainedModels()
      case ERExperimentType.IndependentMulticlassClassifiers => trainIndependentMulticlassClassifiers()
      case ERExperimentType.FactorConstrained => runFactorConstrained()
    }
  }

  object ERExperimentType extends Enumeration {
    val IndependentClassifiers, LPlusI, TestFromModel, JointTraining, PipelineTraining, PipelineTestFromModel, InteractiveMode, IndependentMulticlassClassifiers, FactorConstrained, IndependentClassifiersCV, TestFromModelCV, LPlusICV = Value
  }

  /** in this scenario we train and test classifiers independent of each other. In particular, the relation classifier
    * does not know the labels of its entity arguments, and the entity classifier does not know the labels of relations
    * in the sentence either
    */
  def trainIndependentClassifiers(): Unit = {
    EntityRelationDataModel.populateWithConll()
    val iter = 10
    // independent entity and relation classifiers
    ClassifierUtils.TrainClassifiers(iter, PersonClassifier, OrganizationClassifier, LocationClassifier,
      WorksForClassifier, LivesInClassifier, LocatedInClassifier, OrgBasedInClassifier)
    ClassifierUtils.TestClassifiers(PersonClassifier, OrganizationClassifier, LocationClassifier,
      WorksForClassifier, LivesInClassifier, LocatedInClassifier, OrgBasedInClassifier)
    ClassifierUtils.SaveClassifiers(PersonClassifier, OrganizationClassifier, LocationClassifier,
      WorksForClassifier, LivesInClassifier, LocatedInClassifier, OrgBasedInClassifier)
  }

  private case class RawPrediction(label: String, predicted: String, instance: Option[Any] = None)
  private case class BinaryResult(precision: Double, recall: Double, f1: Double) {
    override def toString: String = f"Precision: ${precision * 100}%.2f // Recall: ${recall * 100}%.2f // F1: ${f1 * 100}%.2f"
  }

  private def getCrossValidationData(numFolds: Int): Seq[(Int, List[ConllRawSentence], List[ConllRawSentence])] = {
    val sentencesAll = EntityRelationSensors.sentencesAll

    val numSentences = sentencesAll.size
    val shuffledSentences = new Random(42).shuffle(sentencesAll)
    val partitions = shuffledSentences.grouped((numSentences + numFolds - 1) / numFolds).toList

    (0 until 5).map({ fold: Int =>
      val trainingSentences = partitions.zipWithIndex.filter(_._2 != fold).flatMap(_._1)
      val testingSentences = partitions(fold)

      (fold, trainingSentences, testingSentences)
    })
  }

  private def getBinaryPredictionResult(predictions: Seq[RawPrediction]): BinaryResult = {
    var truePositive = 0
    var trueNegative = 0
    var falsePositive = 0
    var falseNegative = 0

    predictions.foreach({ prediction =>
      if (prediction.label == "true") {
        prediction.predicted match {
          case "true" => truePositive += 1
          case "false" => falseNegative += 1
        }
      } else {
        prediction.predicted match {
          case "true" => falsePositive += 1
          case "false" => trueNegative += 1
        }
      }
    })

    val precision = truePositive.toDouble / (falsePositive + truePositive)
    val recall = truePositive.toDouble / (falseNegative + truePositive)
    val f1 = (2.0 * precision * recall) / (precision + recall)

    BinaryResult(precision, recall, f1)
  }

  def trainIndependentClassifiersCV(): Unit = {
    val classifiers = Array(PersonClassifier, OrganizationClassifier, LocationClassifier,
      WorksForClassifier, LivesInClassifier, LocatedInClassifier, OrgBasedInClassifier)

    // independent entity and relation classifiers
    val iter = 10

    val allResults = getCrossValidationData(5).flatMap({
      case (fold: Int, trainingSentences: List[ConllRawSentence], testingSentences: List[ConllRawSentence]) =>
        EntityRelationDataModel.clearInstances()
        EntityRelationDataModel.sentences.populate(trainingSentences)
        EntityRelationDataModel.sentences.populate(testingSentences, train = false)

        classifiers.foreach(_.forget())
        classifiers.foreach(_.learn(iter))

        classifiers.map({ clf =>
          clf.modelSuffix = s"fold$fold"
          clf.save()

          val results = clf.node.getTestingInstances.map({ instance =>
            val label = clf.classifier.getLabeler.discreteValue(instance)
            val prediction = clf(instance)

            RawPrediction(label, prediction)
          })

          (clf, fold, results)
        })
    })

    allResults.groupBy(_._1).values.foreach({ singleClassifierResult =>
      val clf = singleClassifierResult.head._1.getClassSimpleNameForClassifier
      val rawPredictions = singleClassifierResult.flatMap(_._3)
      val binaryResult = getBinaryPredictionResult(rawPredictions)
      logger.info(f"$clf // $binaryResult")
    })
  }

  def testIndependentClassifiersCV(): Unit = {
    val classifiers = Array(PersonClassifier, OrganizationClassifier, LocationClassifier,
      WorksForClassifier, LivesInClassifier, LocatedInClassifier, OrgBasedInClassifier)

    val allResults = getCrossValidationData(5)
      .flatMap({
        case (fold: Int, _: List[ConllRawSentence], testingSentences: List[ConllRawSentence]) =>
          EntityRelationDataModel.clearInstances()
          EntityRelationDataModel.sentences.populate(testingSentences, train = false)

          classifiers.foreach(_.forget())

          classifiers.map({ clf =>
            clf.modelSuffix = s"fold$fold"
            clf.load()

            val results = clf.node.getTestingInstances.map({ instance =>
              val label = clf.classifier.getLabeler.discreteValue(instance)
              val prediction = clf(instance)

              RawPrediction(label, prediction)
            })

            (clf, fold, results)
          })
      })

    allResults.groupBy(_._1).values.foreach({ singleClassifierResult =>
      val clf = singleClassifierResult.head._1.getClassSimpleNameForClassifier
      singleClassifierResult.groupBy(_._2).foreach({
        case (fold: Int, items) =>
          val rawPredictions = items.flatMap(_._3)
          val binaryResult = getBinaryPredictionResult(rawPredictions)
          logger.info(f"$clf // Fold: $fold // $binaryResult")
      })
    })
  }

  def runLPlusICV(): Unit = {
    val classifiers = List(PerConstrainedClassifier, OrgConstrainedClassifier, LocConstrainedClassifier,
      WorksForRelationConstrainedClassifier, LivesInRelationConstrainedClassifier)

    val allResults = getCrossValidationData(5)
      .flatMap({
        case (fold: Int, _: List[ConllRawSentence], testingSentences: List[ConllRawSentence]) =>
          EntityRelationDataModel.clearInstances()
          EntityRelationDataModel.sentences.populate(testingSentences, train = false)

          val onClassifiers = classifiers.map(_.onClassifier.asInstanceOf[Learnable[_]]).distinct
          onClassifiers.foreach({ clf =>
            clf.forget()
            clf.modelSuffix = s"fold$fold"
            clf.load()
          })

          classifiers.map({ constrainedClassifier =>
            val clf = constrainedClassifier.onClassifier.asInstanceOf[Learnable[_]]
            val results = clf.node.getTestingInstances.map({ instance =>
              val typedInstance: constrainedClassifier.InstanceType = instance.asInstanceOf[constrainedClassifier.InstanceType]
              val label = clf.classifier.getLabeler.discreteValue(typedInstance)
              val prediction = constrainedClassifier(typedInstance)
              RawPrediction(label, prediction)
            })

            (constrainedClassifier, fold, results)
          })
      })

    allResults.groupBy(_._1).values.foreach({ singleClassifierResult =>
      val clf = singleClassifierResult.head._1.getClassSimpleNameForClassifier
      singleClassifierResult.groupBy(_._2).foreach({
        case (fold: Int, items) =>
          val rawPredictions = items.flatMap(_._3)
          val binaryResult = getBinaryPredictionResult(rawPredictions)
          logger.info(f"$clf // Fold: $fold // $binaryResult")
      })
    })
  }

  /** This function loads the classifiers trained in function [[trainIndependentClassifiers]] and evaluates on the
    * test data.
    */
  def testIndependentClassifiers(): Unit = {
    EntityRelationDataModel.populateWithConll()
    ClassifierUtils.LoadClassifier(
      jarModelPath,
      PersonClassifier, OrganizationClassifier, LocationClassifier,
      WorksForClassifier, LivesInClassifier, LocatedInClassifier, OrgBasedInClassifier
    )
    ClassifierUtils.TestClassifiers(PersonClassifier, OrganizationClassifier, LocationClassifier,
      WorksForClassifier, LivesInClassifier, LocatedInClassifier, OrgBasedInClassifier)
  }

  /** in this scenario the named entity recognizers are trained independently, and given to a relation classifier as
    * a tool to extract features (hence the name "pipeline"). This approach first trains an entity classifier, and
    * then uses the prediction of entities in addition to other local features to learn the relation identifier.
    */
  def runPipelineTraining(): Unit = {
    EntityRelationDataModel.populateWithConll()
    ClassifierUtils.LoadClassifier(jarModelPath, PersonClassifier, OrganizationClassifier, LocationClassifier)

    // train pipeline relation models, which use the prediction of the entity classifiers
    val iter = 10
    ClassifierUtils.TrainClassifiers(iter, WorksForClassifierPipeline, LivesInClassifierPipeline)
    ClassifierUtils.TestClassifiers(WorksForClassifierPipeline, LivesInClassifierPipeline)
    ClassifierUtils.SaveClassifiers(WorksForClassifierPipeline, LivesInClassifierPipeline)
  }

  /** this function loads the models of the pipeline classifiers and evaluates them on the test data */
  def testPipelineRelationModels(): Unit = {
    EntityRelationDataModel.populateWithConll()
    ClassifierUtils.LoadClassifier(jarModelPath, PersonClassifier, OrganizationClassifier, LocationClassifier,
      WorksForClassifierPipeline, LivesInClassifierPipeline)
    ClassifierUtils.TestClassifiers(WorksForClassifierPipeline, LivesInClassifierPipeline)
  }

  /** In the scenario the classifiers are learned independently but at the test time we use constrained inference to
    * maintain structural consistency (which would justify the naming "Learning Plus Inference" (L+I).
    */
  def runLPlusI() {
    EntityRelationDataModel.populateWithConll()

    // load all independent models
    ClassifierUtils.LoadClassifier(jarModelPath, PersonClassifier, OrganizationClassifier, LocationClassifier,
      WorksForClassifier, LivesInClassifier, LocatedInClassifier, OrgBasedInClassifier)

    //     Test using constrained classifiers
    ClassifierUtils.TestClassifiers(PerConstrainedClassifier, OrgConstrainedClassifier, LocConstrainedClassifier,
      WorksForRelationConstrainedClassifier, LivesInRelationConstrainedClassifier)
  }

  /** here we meanwhile training classifiers, we use global inference, in order to overcome the poor local
    * classifications and yield accurate global classifications.
    */
  def runJointTraining() {
    populateWithConll()
    val testRels = pairs.getTestingInstances.toSet.toList
    val testTokens = tokens.getTestingInstances.toSet.toList

    // load pre-trained independent models, the following lines (loading pre-trained models) are not necessary,
    // although without pre-training the performance might drop.
    ClassifierUtils.LoadClassifier(jarModelPath, PersonClassifier, OrganizationClassifier, LocationClassifier,
      WorksForClassifier, LivesInClassifier, LocatedInClassifier, OrgBasedInClassifier)

    // joint training
    val jointTrainIteration = 5
    logger.info(s"Joint training $jointTrainIteration iterations. ")
    JointTrainSparseNetwork.train[ConllRelation](
      pairs,
      PerConstrainedClassifier :: OrgConstrainedClassifier :: LocConstrainedClassifier ::
        WorksForRelationConstrainedClassifier :: LivesInRelationConstrainedClassifier :: Nil,
      jointTrainIteration, init = true
    )

    // TODO: merge the following two tests
    ClassifierUtils.TestClassifiers((testTokens, PerConstrainedClassifier), (testTokens, OrgConstrainedClassifier),
      (testTokens, LocConstrainedClassifier))

    ClassifierUtils.TestClassifiers(
      (testRels, WorksForRelationConstrainedClassifier),
      (testRels, LivesInRelationConstrainedClassifier)
    )
  }

  /** Interactive model to annotate input sentences with Pre-trained models
    */
  def interactiveWithPretrainedModels(): Unit = {
    // Load independent classifiers.
    ClassifierUtils.LoadClassifier(
      jarModelPath,
      PersonClassifier, OrganizationClassifier, LocationClassifier,
      WorksForClassifier, LivesInClassifier, LocatedInClassifier, OrgBasedInClassifier
    )

    val posAnnotator = POSTaggerApp.getPretrainedAnnotator()
    val entityAnnotator = new EntityAnnotator(ViewNames.NER_CONLL)
    val taBuilder = new TokenizerTextAnnotationBuilder(new StatefulTokenizer())

    while (true) {
      println("Enter a sentence to annotate (or Press Enter to exit)")
      val input = StdIn.readLine()

      input match {
        case sentence: String if sentence.trim.nonEmpty =>
          // Create a Text Annotation with the current input sentence.
          val ta = taBuilder.createTextAnnotation(sentence.trim)
          posAnnotator.addView(ta)
          entityAnnotator.addView(ta)

          println("Part-Of-Speech View: " + ta.getView(ViewNames.POS).toString)
          println("Entity View: " + ta.getView(ViewNames.NER_CONLL).toString)
        case _ => return
      }
    }
  }

  /* Independent Multiclass Classifiers */
  def trainIndependentMulticlassClassifiers(): Unit = {
    EntityRelationDataModel.populateWithConll()
    val iter = 10

    // independent entity and relation classifiers
    ClassifierUtils.TrainClassifiers(iter, EntityMulticlassClassifier, RelationMulticlassClassifier)
    EntityMulticlassClassifier.test()

    // The Kill relation is not present in the training set.
    val filteredTestRelations = pairs.getTestingInstances.filter(_.relType != "Kill")
    RelationMulticlassClassifier.test(filteredTestRelations)
    ClassifierUtils.SaveClassifiers(EntityMulticlassClassifier, RelationMulticlassClassifier)
  }

  def runFactorConstrained(): Unit = {
    populateWithConll()

    // load pre-trained independent models, the following lines (loading pre-trained models) are not necessary,
    // although without pre-training the performance might drop.
    ClassifierUtils.LoadClassifier(jarModelPath, PersonClassifier, OrganizationClassifier, LocationClassifier,
      WorksForClassifier, LivesInClassifier, LocatedInClassifier, OrgBasedInClassifier)

    val filteredTestRelations = pairs.getTestingInstances.filter(_.relType != "Kill")
    WorksForClassifier.test(filteredTestRelations)
    WorksForRelationConstrainedClassifier.test(filteredTestRelations)

    LivesInClassifier.test(filteredTestRelations)
    LivesInRelationConstrainedClassifier.test(filteredTestRelations)
  }
}
