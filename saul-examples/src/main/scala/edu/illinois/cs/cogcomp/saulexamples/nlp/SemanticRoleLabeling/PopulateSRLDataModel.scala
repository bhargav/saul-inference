/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling

import java.util.Properties

import edu.illinois.cs.cogcomp.annotation.AnnotatorException
import edu.illinois.cs.cogcomp.core.datastructures.ViewNames
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.{ Constituent, TextAnnotation, TreeView }
import edu.illinois.cs.cogcomp.core.datastructures.trees.Tree
import edu.illinois.cs.cogcomp.curator.CuratorConfigurator._
import edu.illinois.cs.cogcomp.edison.annotators.ClauseViewGenerator
import edu.illinois.cs.cogcomp.nlp.common.PipelineConfigurator._
import edu.illinois.cs.cogcomp.nlp.utilities.ParseUtils
import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.illinois.cs.cogcomp.saulexamples.data.SRLDataReader
import edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling.SRLSensors._
import edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling.SRLscalaConfigurator._
import edu.illinois.cs.cogcomp.saulexamples.nlp.TextAnnotationFactory

import scala.collection.JavaConversions._

import SRLMultiGraphDataModel._

/** Created by Parisa on 1/17/16.
  */
object PopulateSRLDataModel extends Logging {
  def apply[T <: AnyRef](
    testOnly: Boolean = false,
    useGoldPredicate: Boolean = false,
    useGoldArgBoundaries: Boolean = false
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
        TextAnnotationFactory.disableSettings(nonDefaultProps, USE_NER_CONLL, USE_NER_ONTONOTES, USE_SRL_VERB, USE_SRL_NOM, USE_STANFORD_DEP)
        if (parseViewName.equals(ViewNames.PARSE_GOLD))
          TextAnnotationFactory.disableSettings(nonDefaultProps, USE_POS, USE_STANFORD_PARSE)
        TextAnnotationFactory.createPipelineAnnotatorService(nonDefaultProps)
    }

    val clauseViewGenerator = parseViewName match {
      case ViewNames.PARSE_GOLD => new ClauseViewGenerator(parseViewName, "CLAUSES_GOLD")
      case ViewNames.PARSE_STANFORD => ClauseViewGenerator.STANFORD
    }

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
          case e: AnnotatorException =>
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

    def populateDocument(a: TextAnnotation, isTrainingInstance: Boolean): Unit = {
      if (!useGoldPredicate) {
        sentences.populate(Seq(a), train = isTrainingInstance)

        val predicateTrainCandidates = (sentences(a) ~> sentencesToTokens).collect({
          case x: Constituent if posTag(x).startsWith("VB") => x.cloneForNewView(ViewNames.SRL_VERB)
        })

        predicates.populate(predicateTrainCandidates, train = isTrainingInstance)
      } else {
        sentences.populate(Seq(a), train = isTrainingInstance)
      }
      logger.debug("gold relations for this train:" + (sentences(a) ~> sentencesToRelations).size)

      if (!useGoldArgBoundaries) {
        val XuPalmerCandidateArgsTraining = (sentences(a) ~> sentencesToRelations ~> relationsToPredicates).flatMap({
          x => xuPalmerCandidate(x, (sentences(x.getTextAnnotation) ~> sentencesToStringTree).head)
        })

        relations.populate(XuPalmerCandidateArgsTraining, train = isTrainingInstance)
      }

      logger.debug("all relations for this test:" + (sentences(a) ~> sentencesToRelations).size)

      if (sentences().size % 1000 == 0) logger.info("loaded graphs in memory:" + sentences().size)
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
