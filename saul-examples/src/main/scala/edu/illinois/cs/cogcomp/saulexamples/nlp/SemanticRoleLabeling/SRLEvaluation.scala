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
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.{ TextAnnotation, TreeView }
import edu.illinois.cs.cogcomp.core.datastructures.trees.Tree
import edu.illinois.cs.cogcomp.core.experiments.ClassificationTester
import edu.illinois.cs.cogcomp.core.experiments.evaluators.PredicateArgumentEvaluator
import edu.illinois.cs.cogcomp.core.utilities.protobuf.ProtobufSerializer
import edu.illinois.cs.cogcomp.curator.CuratorConfigurator.RESPECT_TOKENIZATION
import edu.illinois.cs.cogcomp.nlp.utilities.ParseUtils
import edu.illinois.cs.cogcomp.pipeline.common.PipelineConfigurator.{ USE_LEMMA, USE_POS, USE_SHALLOW_PARSE, USE_STANFORD_PARSE }
import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.illinois.cs.cogcomp.saulexamples.data.SRLDataReader
import edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling.SRLscalaConfigurator.{ PROPBANK_HOME, TEST_SECTION, TREEBANK_HOME }
import edu.illinois.cs.cogcomp.saulexamples.nlp.TextAnnotationFactory
import org.mapdb.{ DB, DBMaker, Serializer }

import scala.collection.JavaConverters._

/** Evaluate the SRL Annotator using PredicateArgumentEvaluator.
  * This evaluation honors settings in the SRLscalaConfigurator class.
  */
object SRLEvaluation extends App with Logging {
  val parseViewName = SRLscalaConfigurator.SRL_PARSE_VIEW
  val predictedViewName = ViewNames.SRL_VERB + "_PREDICTED"
  val annotator = new SRLAnnotator(predictedViewName)
  val sourceDatasetWithPrerequisites = s"Curator=${SRLscalaConfigurator.USE_CURATOR}_${parseViewName}_${TEST_SECTION}.cache"
  val annotatedDatasetCache = s"${predictedViewName}_${sourceDatasetWithPrerequisites}"
  private var databaseInstance: Option[DB] = None

  logger.info(s"Initializing the annotator service: USE_CURATOR = ${SRLscalaConfigurator.USE_CURATOR}")
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

  val cachedDataset = fetchDatasetFromCache(sourceDatasetWithPrerequisites)
  val preProcessedDocuments = {
    if (cachedDataset.nonEmpty) {
      cachedDataset
    } else {
      val viewsToKeep = Set(ViewNames.TOKENS, ViewNames.SENTENCE, ViewNames.SRL_VERB, ViewNames.PARSE_GOLD)
      logger.info("Moving existing GOLD annotation views")

      val testReader = new SRLDataReader(TREEBANK_HOME, PROPBANK_HOME, TEST_SECTION, TEST_SECTION)

      logger.info("Reading the dataset.")
      testReader.readData()

      val dataset = testReader.textAnnotations.asScala
        .map({ ta =>
          ta.getAvailableViews
            .asScala
            .diff(viewsToKeep)
            .foreach({ viewName =>
              logger.debug(s"Removing view $viewName")
              ta.removeView(viewName)
            })

          ta
        })

      putDatasetInCache(dataset, sourceDatasetWithPrerequisites)
      dataset
    }
  }

  logger.info("Annotating documents with pre-requisite views")
  val annotatedDocumentsPartial = preProcessedDocuments.map({ ta =>
    try {
      // Add new views
      annotatorService.addView(ta, ViewNames.LEMMA)
      annotatorService.addView(ta, ViewNames.SHALLOW_PARSE)

      if (!parseViewName.equals(ViewNames.PARSE_GOLD)) {
        annotatorService.addView(ta, ViewNames.POS)
        annotatorService.addView(ta, parseViewName)
      }

      // Clean up the trees
      val tree: Tree[String] = ta.getView(parseViewName).asInstanceOf[TreeView].getTree(0)
      val parseView = new TreeView(parseViewName, ta)
      parseView.setParseTree(0, ParseUtils.stripFunctionTags(ParseUtils.snipNullNodes(tree)))
      ta.addView(parseViewName, parseView)

      Some(ta)
    } catch {
      case ex: Exception =>
        logger.error(s"Annotation failed for sentence ${ta.getId}; removing it from the list.", ex)
        None
    }
  }).partition(_.isEmpty)

  logger.info("Starting SRL Annotation and evaluation")

  val identifierTester = new ClassificationTester
  identifierTester.ignoreLabelFromSummary("V")

  val evaluator = new PredicateArgumentEvaluator

  // Annotate with SRL Annotator
  var srlAnnotationFailures = 0
  val annotatedDocuments = annotatedDocumentsPartial._2
    .flatten
    .flatMap({ ta =>
      try {
        annotator.addView(ta)
        evaluator.evaluate(identifierTester, ta.getView(ViewNames.SRL_VERB), ta.getView(predictedViewName))
        Some(ta)
      } catch {
        case ex: Exception =>
          srlAnnotationFailures += 1
          logger.error(s"SRL Annotation failed for sentence ${ta.getId}.", ex)
          None
      }
    })

  putDatasetInCache(annotatedDocuments, annotatedDatasetCache)

  logger.info(s"Pipeline/Curator Annotation failures = ${annotatedDocumentsPartial._1.size}")
  logger.info(s"Pipeline/Curator Annotation success = ${annotatedDocumentsPartial._2.size}")
  logger.info(s"USE_CURATOR = ${SRLscalaConfigurator.USE_CURATOR}")
  logger.info(s"Documents which failed SRL Annotation = $srlAnnotationFailures")
  println(identifierTester.getPerformanceTable(true).toTextTable)

  def putDatasetInCache(dataset: Seq[TextAnnotation], datasetName: String): Unit = {
    openDatabase(datasetName)

    val datasetMap = databaseInstance.map({ dataset: DB =>
      dataset.hashMap(datasetName, Serializer.INTEGER, Serializer.BYTE_ARRAY).createOrOpen()
    }).get

    datasetMap.clear()
    val datasetHashMap = dataset.map({ ta: TextAnnotation =>
      val hashCode = ta.getTokenizedText.hashCode
      new Integer(hashCode) -> ProtobufSerializer.writeAsBytes(ta)
    }).toMap

    datasetMap.putAll(datasetHashMap.asJava)
  }

  def fetchDatasetFromCache(datasetName: String): Seq[TextAnnotation] = {
    try {
      openDatabase(datasetName)

      databaseInstance.map({ dataset: DB =>
        dataset.hashMap(datasetName, Serializer.INTEGER, Serializer.BYTE_ARRAY).createOrOpen()
      }).map({ dataMap =>
        dataMap.asScala.map({ case (_: Integer, taBytes: Array[Byte]) => ProtobufSerializer.parseFrom(taBytes) })
      }).map(_.toSeq)
        .getOrElse(Seq.empty)
    } catch {
      case _: Exception =>
        logger.error("Error while reading cache file")
        Seq.empty
    }
  }

  def openDatabase(datasetName: String): Unit = {
    if (databaseInstance.nonEmpty) {
      closeDatabase()
    }

    databaseInstance = Some(DBMaker.fileDB(datasetName)
      .closeOnJvmShutdown()
      .make())
  }

  def closeDatabase(): Unit = {
    if (databaseInstance.nonEmpty) {
      databaseInstance.get.close()
      databaseInstance = None
    }
  }
}
