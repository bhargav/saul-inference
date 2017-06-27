/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling

import java.util.Properties

import edu.illinois.cs.cogcomp.annotation.{ AnnotatorException, AnnotatorServiceConfigurator }
import edu.illinois.cs.cogcomp.core.datastructures.ViewNames
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.{ Constituent, TextAnnotation, TreeView }
import edu.illinois.cs.cogcomp.core.datastructures.trees.Tree
import edu.illinois.cs.cogcomp.curator.CuratorConfigurator._
import edu.illinois.cs.cogcomp.edison.annotators.ClauseViewGenerator
import edu.illinois.cs.cogcomp.pipeline.common.PipelineConfigurator._
import edu.illinois.cs.cogcomp.nlp.utilities.ParseUtils
import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.illinois.cs.cogcomp.saulexamples.data.SRLDataReader
import edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling.SRLSensors._
import edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling.SRLscalaConfigurator._
import edu.illinois.cs.cogcomp.saulexamples.nlp.TextAnnotationFactory

import scala.collection.JavaConversions._

/** Created by Parisa on 1/17/16.
  */
object PopulateSRLDataModel extends Logging {
  def apply[T <: AnyRef](
    testOnly: Boolean = false,
    useGoldPredicate: Boolean = false,
    useGoldArgBoundaries: Boolean = false,
    usePipelineCaching: Boolean = true
  ): Unit = {

    val useCurator = SRLscalaConfigurator.USE_CURATOR
    val parseViewName = SRLscalaConfigurator.SRL_PARSE_VIEW

    val annotatorService = useCurator match {
      case true =>
        val nonDefaultProps = new Properties()
        TextAnnotationFactory.enableSettings(nonDefaultProps, RESPECT_TOKENIZATION)
        TextAnnotationFactory.createCuratorAnnotatorService(nonDefaultProps)
      case false =>
        val nonDefaultProps = new Properties()
        TextAnnotationFactory.enableSettings(nonDefaultProps, USE_LEMMA, USE_SHALLOW_PARSE)
        if (!parseViewName.equals(ViewNames.PARSE_GOLD)) {
          TextAnnotationFactory.enableSettings(nonDefaultProps, USE_POS, USE_STANFORD_PARSE)
        }
        if (!usePipelineCaching) {
          TextAnnotationFactory.enableSettings(nonDefaultProps, AnnotatorServiceConfigurator.DISABLE_CACHE)
        }
        TextAnnotationFactory.createPipelineAnnotatorService(nonDefaultProps)
    }

    val clauseViewGenerator = parseViewName match {
      case ViewNames.PARSE_GOLD => new ClauseViewGenerator(parseViewName, "CLAUSES_GOLD")
      case ViewNames.PARSE_STANFORD => ClauseViewGenerator.STANFORD
    }

    /**
      * Add required views to the text annotations and filter out failed text annotations.
      *
      * @param taAll Input text annotations.
      * @return Text annotations with required views populated.s
      */
    def addViewAndFilter(taAll: Iterable[TextAnnotation]): Iterable[TextAnnotation] = {
      taAll.flatMap({ ta =>
        try {
          annotatorService.addView(ta, ViewNames.LEMMA)
          annotatorService.addView(ta, ViewNames.SHALLOW_PARSE)
          if (!parseViewName.equals(ViewNames.PARSE_GOLD)) {
            annotatorService.addView(ta, ViewNames.POS)
            annotatorService.addView(ta, ViewNames.PARSE_STANFORD)
          }
          // Add a clause view (needed for the clause relative position feature)
          clauseViewGenerator.addView(ta)

          // Clean up the trees
          val tree: Tree[String] = ta.getView(parseViewName).asInstanceOf[TreeView].getTree(0)
          val parseView = new TreeView(parseViewName, ta)
          parseView.setParseTree(0, ParseUtils.stripFunctionTags(ParseUtils.snipNullNodes(tree)))
          ta.addView(parseViewName, parseView)

          Some(ta)
        } catch {
          case e: Exception =>
            logger.warn(s"Annotation failed for sentence ${ta.getId}; removing it from the list.")
            None
        }
      })
    }

    def printNumbers(reader: SRLDataReader, readerType: String) = {
      val numPredicates = reader.textAnnotations.map(ta => ta.getView(ViewNames.SRL_VERB).getConstituents.count(c => c.getLabel == "Predicate")).sum
      val numArguments = reader.textAnnotations.map(ta => ta.getView(ViewNames.SRL_VERB).getConstituents.count(c => c.getLabel != "Predicate")).sum
      logger.debug(s"Number of $readerType data predicates: $numPredicates")
      logger.debug(s"Number of $readerType data arguments: $numArguments")
    }

    /**
      * Adds a single Text Annotation to the DataModel graph.
      *
      * @param a Text Annotation instance to add to the graph.s
      * @param isTrainingInstance Boolean indicating if the instance is a training instance.s
      */
    def populateDocument(a: TextAnnotation, isTrainingInstance: Boolean): Unit = {
      // Data Model graph for a single sentence
      val singletonGraph = new SRLMultiGraphDataModel(parseViewName)

      // Populate the sentence node.
      // Note: This does not populate the relation/predicates/arguments node  s.
      singletonGraph.sentences.populate(Seq(a), train = isTrainingInstance)

      if (!useGoldPredicate) {


        val predicateTrainCandidates = (singletonGraph.sentences(a) ~> singletonGraph.sentencesToTokens).collect({
          case x: Constituent if singletonGraph.posTag(x).startsWith("VB") => x.cloneForNewView(ViewNames.SRL_VERB)
        })

        singletonGraph.predicates.populate(predicateTrainCandidates, train = isTrainingInstance)
      } else {
        singletonGraph.sentences.populate(Seq(a), train = isTrainingInstance)
      }
      logger.debug("gold relations for this train:" + (singletonGraph.sentences(a) ~> singletonGraph.sentencesToRelations).size)

      if (!useGoldArgBoundaries) {
        val XuPalmerCandidateArgsTraining = (singletonGraph.sentences(a) ~> singletonGraph.sentencesToRelations ~> singletonGraph.relationsToPredicates).flatMap({
          x => xuPalmerCandidate(x, (singletonGraph.sentences(x.getTextAnnotation) ~> singletonGraph.sentencesToStringTree).head)
        })

        singletonGraph.relations.populate(XuPalmerCandidateArgsTraining, train = isTrainingInstance)
      }

      logger.debug("all relations for this test:" + (singletonGraph.sentences(a) ~> singletonGraph.sentencesToRelations).size)

      SRLClassifiers.SRLDataModel.addFromModel(singletonGraph)
      if (singletonGraph.sentences().size % 1000 == 0) logger.info("loaded graphs in memory:" + singletonGraph.sentences().size)
    }

    if (!testOnly) {
      logger.info(s"Reading training data from sections $TRAIN_SECTION_S to $TRAIN_SECTION_E")
      val trainReader = new SRLDataReader(TREEBANK_HOME, PROPBANK_HOME,
        TRAIN_SECTION_S, TRAIN_SECTION_E)
      trainReader.readData()
      logger.info(s"Annotating ${trainReader.textAnnotations.size} training sentences")
      val filteredTa = addViewAndFilter(trainReader.textAnnotations)
      printNumbers(trainReader, "training")
      logger.info("Populating SRLDataModel with training data.")

      filteredTa.foreach(populateDocument(_, isTrainingInstance = true))
    }

    val testReader = new SRLDataReader(TREEBANK_HOME, PROPBANK_HOME, TEST_SECTION, TEST_SECTION)
    logger.info(s"Reading test data from section $TEST_SECTION")
    testReader.readData()

    logger.info(s"Annotating ${testReader.textAnnotations.size} test sentences")
    val filteredTest = addViewAndFilter(testReader.textAnnotations)

    printNumbers(testReader, "test")

    logger.info("Populating SRLDataModel with test data.")
    filteredTest.foreach(populateDocument(_, isTrainingInstance = false))
  }
}
