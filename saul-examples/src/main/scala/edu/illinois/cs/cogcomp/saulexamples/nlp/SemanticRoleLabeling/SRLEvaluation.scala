/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling

import java.util.Properties

import edu.illinois.cs.cogcomp.annotation.AnnotatorServiceConfigurator
import edu.illinois.cs.cogcomp.core.datastructures.ViewNames
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.TreeView
import edu.illinois.cs.cogcomp.core.datastructures.trees.Tree
import edu.illinois.cs.cogcomp.core.experiments.ClassificationTester
import edu.illinois.cs.cogcomp.core.experiments.evaluators.PredicateArgumentEvaluator
import edu.illinois.cs.cogcomp.curator.CuratorConfigurator.RESPECT_TOKENIZATION
import edu.illinois.cs.cogcomp.nlp.utilities.ParseUtils
import edu.illinois.cs.cogcomp.pipeline.common.PipelineConfigurator.{ USE_LEMMA, USE_POS, USE_SHALLOW_PARSE, USE_STANFORD_PARSE }
import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.illinois.cs.cogcomp.saulexamples.data.SRLDataReader
import edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling.SRLscalaConfigurator.{ PROPBANK_HOME, TEST_SECTION, TREEBANK_HOME }
import edu.illinois.cs.cogcomp.saulexamples.nlp.TextAnnotationFactory

import scala.collection.JavaConverters._

/** Evaluate the SRL Annotator using PredicateArgumentEvaluator.
  * This evaluation honors settings in the SRLscalaConfigurator class.
  */
object SRLEvaluation extends App with Logging {
  val parseViewName = SRLscalaConfigurator.SRL_PARSE_VIEW
  val predictedViewName = ViewNames.SRL_VERB + "_PREDICTED"
  val annotator = new SRLAnnotator(predictedViewName)

  val testReader = new SRLDataReader(TREEBANK_HOME, PROPBANK_HOME, TEST_SECTION, TEST_SECTION)

  logger.info("Reading the dataset.")
  testReader.readData()

  logger.info(s"Intializing the annotator service: USE_CURATOR = ${SRLscalaConfigurator.USE_CURATOR}")
  val usePipelineCaching = true
  val annotatorService = SRLscalaConfigurator.USE_CURATOR match {
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

  logger.info("Annotating documents with pre-requisite views")
  val annotatedDocumentsPartial = testReader.textAnnotations.asScala.map({ ta =>
    try {
      annotatorService.addView(ta, ViewNames.LEMMA)
      annotatorService.addView(ta, ViewNames.SHALLOW_PARSE)
      if (!parseViewName.equals(ViewNames.PARSE_GOLD)) {
        annotatorService.addView(ta, ViewNames.POS)
        annotatorService.addView(ta, ViewNames.PARSE_STANFORD)
      }

      // Clean up the trees
      val tree: Tree[String] = ta.getView(parseViewName).asInstanceOf[TreeView].getTree(0)
      val parseView = new TreeView(parseViewName, ta)
      parseView.setParseTree(0, ParseUtils.stripFunctionTags(ParseUtils.snipNullNodes(tree)))
      ta.addView(parseViewName, parseView)

      Some(ta)
    } catch {
      case _: Exception =>
        logger.warn(s"Annotation failed for sentence ${ta.getId}; removing it from the list.")
        None
    }
  }).partition(_.isEmpty)

  logger.info(s"Annotation failures = ${annotatedDocumentsPartial._1.size}")
  logger.info(s"Annotation success = ${annotatedDocumentsPartial._2.size}")
  logger.info("Starting SRL Annotation and evaluation")

  val identifierTester = new ClassificationTester
  val evaluator = new PredicateArgumentEvaluator

  // Annotate with SRL Annotator
  var srlAnnotationFailures = 0
  annotatedDocumentsPartial._2
    .flatten
    .foreach({ ta =>
      try {
        annotator.addView(ta)
        evaluator.evaluate(identifierTester, ta.getView(ViewNames.SRL_VERB), ta.getView(predictedViewName))
      } catch {
        case _: Exception =>
          srlAnnotationFailures += 1
          logger.warn(s"SRL Annotation failed for sentence ${ta.getId}.")
      }
    })

  logger.info(s"Documents which failed SRL Annotation = $srlAnnotationFailures")
  println(identifierTester.getPerformanceTable(true).toTextTable)
}
