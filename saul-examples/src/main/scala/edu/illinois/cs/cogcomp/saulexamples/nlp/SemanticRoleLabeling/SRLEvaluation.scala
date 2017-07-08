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
import edu.illinois.cs.cogcomp.core.datastructures.textannotation._
import edu.illinois.cs.cogcomp.core.datastructures.trees.Tree
import edu.illinois.cs.cogcomp.core.experiments.ClassificationTester
import edu.illinois.cs.cogcomp.core.experiments.evaluators.PredicateArgumentEvaluator
import edu.illinois.cs.cogcomp.core.utilities.protobuf.ProtobufSerializer
import edu.illinois.cs.cogcomp.curator.CuratorConfigurator
import edu.illinois.cs.cogcomp.nlp.utilities.ParseUtils
import edu.illinois.cs.cogcomp.pipeline.common.PipelineConfigurator.{ USE_LEMMA, USE_POS, USE_SHALLOW_PARSE, USE_STANFORD_PARSE }
import edu.illinois.cs.cogcomp.saul.util.Logging
import edu.illinois.cs.cogcomp.saulexamples.data.SRLDataReader
import edu.illinois.cs.cogcomp.saulexamples.nlp.SemanticRoleLabeling.SRLscalaConfigurator.{ PROPBANK_HOME, TEST_SECTION, TREEBANK_HOME }
import edu.illinois.cs.cogcomp.saulexamples.nlp.TextAnnotationFactory
import org.mapdb.{ DB, DBMaker, Serializer }

import scala.collection.JavaConverters._
import scala.collection.mutable

/** Evaluate the SRL Annotator using PredicateArgumentEvaluator.
  * This evaluation honors settings in the SRLscalaConfigurator class.
  */
object SRLEvaluation extends App with Logging {
  val parseViewName = SRLscalaConfigurator.SRL_PARSE_VIEW
  val predictedViewName = ViewNames.SRL_VERB + "_PREDICTED_BEAM_10"
  val srl_model_name = "verb-stanford"
  val annotator = new SRLAnnotator(predictedViewName)
  val sourceDatasetWithPrerequisites = s"Curator=${SRLscalaConfigurator.USE_CURATOR}_${parseViewName}_${TEST_SECTION}.cache"
  val annotatedDatasetCache = sourceDatasetWithPrerequisites // s"${srl_model_name}_${sourceDatasetWithPrerequisites}"
  var databaseInstance: Option[DB] = None
  val evaluateConstraints = true

  if (!evaluateConstraints) {
    logger.info(s"Initializing the annotator service: USE_CURATOR = ${SRLscalaConfigurator.USE_CURATOR}")
    val usePipelineCaching = true
    lazy val annotatorService = SRLscalaConfigurator.USE_CURATOR match {
      case true =>
        val nonDefaultProps = new Properties()
        TextAnnotationFactory.enableSettings(nonDefaultProps, CuratorConfigurator.RESPECT_TOKENIZATION)
        // TextAnnotationFactory.enableSettings(nonDefaultProps, CuratorConfigurator.CURATOR_FORCE_UPDATE)
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

    logger.info("Trying to read dataset from cache file.")
    val cachedDataset = fetchDatasetFromCache(annotatedDatasetCache)
    val preProcessedDocuments = {
      if (cachedDataset.nonEmpty) {
        logger.info("Reading dataset from cache")
        cachedDataset.map(Some(_))
      } else {
        val viewsToKeep = Set(ViewNames.TOKENS, ViewNames.SENTENCE, ViewNames.SRL_VERB, ViewNames.PARSE_GOLD)
        val testReader = new SRLDataReader(TREEBANK_HOME, PROPBANK_HOME, TEST_SECTION, TEST_SECTION)

        logger.info("Reading the dataset.")
        testReader.readData()

        logger.info("Annotating documents with pre-requisite views")
        val dataset = testReader.textAnnotations.asScala
          .map({ ta =>
            ta.getAvailableViews
              .asScala
              .diff(viewsToKeep)
              .foreach({ viewName =>
                logger.debug(s"Removing view $viewName")
                ta.removeView(viewName)
              })

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
          })

        putDatasetInCache(dataset.flatten, annotatedDatasetCache)
        dataset
      }
    }

    val annotatedDocumentsPartial = preProcessedDocuments.partition(_.isEmpty)
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

          // logger.info(ta.getView(ViewNames.POS).toString)
          // logger.info(ta.getView(ViewNames.SRL_VERB).toString)
          // logger.info(ta.getView(predictedViewName).toString)

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
    logger.info(identifierTester.getPerformanceTable(true).toTextTable)
  } else {
    val evaluationViewName = "SRL_VERB_PREDICTED_BEAM_5" // ViewNames.SRL_VERB
    val goldViewName = ViewNames.SRL_VERB

    // Evaluate constraints from available views only
    val annotatedDataset = fetchDatasetFromCache(annotatedDatasetCache)
      .partition(_.hasView(evaluationViewName))

    val countList = new mutable.ListBuffer[String]()
    val availableViews = new mutable.HashSet[String]()

    annotatedDataset._1
      .foreach({ document: TextAnnotation =>
        val predictedView = document.getView(evaluationViewName).asInstanceOf[PredicateArgumentView]
        val goldView = document.getView(goldViewName).asInstanceOf[PredicateArgumentView]
        availableViews ++= document.getAvailableViews.asScala

        val predicates = predictedView.getPredicates.asScala.filter({ pred =>
          goldView.getPredicates.asScala.exists({ goldPred => goldPred.getStartSpan == pred.getStartSpan && goldPred.getEndSpan == pred.getEndSpan })
        })
        predicates.foreach({ predicate: Constituent =>
          countList += "Verb"

          val predicateLemma = document.getView(ViewNames.LEMMA).getConstituentsCovering(predicate).get(0).getLabel
          val arguments = predictedView.getArguments(predicate)

          if (arguments.asScala.size == 0) {
            countList += "EmptyArgs"
          } else {
            val labelArgMap = arguments.asScala.groupBy(x => x.getRelationName)
            val argMaxEnd = arguments.asScala.map(_.getTarget.getEndSpan).max + 1
            val occupiedBitSet = new mutable.BitSet(argMaxEnd)

            // No-Overlap constraint violation
            val overlapTrue = arguments.asScala
              .forall({ rel =>
                val range = Range(rel.getTarget.getStartSpan, rel.getTarget.getEndSpan)
                if (range.forall(occupiedBitSet.contains)) {
                  false
                } else {
                  range.forall(occupiedBitSet.add)
                  true
                }
              })
            countList += s"NoOverlap:$overlapTrue"

            // Legal Arguments Constraint
            val legalArgsForVerbOrig = SRLscalaConfigurator.SRL_FRAME_MANAGER.getLegalArguments(predicateLemma).asScala
            val legalArgsForVerb = legalArgsForVerbOrig.union(Set("C-V"))

            val legalArgsTrue = labelArgMap.keySet.diff(legalArgsForVerb).size == 0
            countList += s"LegalArgs:$legalArgsTrue"

            // No duplicate Core Argument
            val coreArgument = List("A0", "A1", "A2", "A3", "A4", "A5", "AA")
            val coreArgsTrue = coreArgument.forall({ coreLabel =>
              labelArgMap.get(coreLabel).forall(_.size <= 1)
            })
            countList += s"CoreArgs:$coreArgsTrue"

            if (coreArgsTrue == false) {
              logger.info(s"Core Args constraint failed at ${predicate.toString}")
              logger.info(predictedView.toString)
            }

            // Referential Argument
            val referentialArguments = Set("R-A1", "R-A2", "R-A3", "R-A4", "R-AA", "R-AM-ADV", "R-AM-CAU", "R-AM-EXT", "R-AM-LOC", "R-AM-MNR", "R-AM-PNC")
            val rArgs = arguments.asScala.filter(x => referentialArguments.contains(x.getRelationName))
            val rArgsTrue = rArgs.forall({ rel =>
              val relLabel = rel.getRelationName.substring(2)
              labelArgMap.contains(relLabel)
            })
            countList += s"RArgs:$rArgsTrue"

            if (rArgsTrue == false) {
              logger.info(s"RArgs failure for ${predicate.toString}")
              logger.info(predictedView.toString)
            }

            // Continuation Argument
            val continuationArguments = Set("C-A1", "C-A2", "C-A3", "C-A4", "C-A5", "C-AM-DIR", "C-AM-LOC", "C-AM-MNR", "C-AM-NEG", "C-AM-PNC")
            val cArgs = arguments.asScala.filter(x => continuationArguments.contains(x.getRelationName))
            val cArgsTrue = cArgs.forall({ rel =>
              val relLabel = rel.getRelationName.substring(2)
              labelArgMap.contains(relLabel)
            })
            countList += s"CArgs:$cArgsTrue"

            if (cArgsTrue == false) {
              logger.info(s"CArgs failure for predicate: ${predicate.toString}")
              logger.info(labelArgMap.toString)
              logger.info(predictedView.toString)
              logger.info("Gold view")
              logger.info(goldView.toString)
            }

            val cArgsOrder = cArgs.forall({ rel =>
              val relLabel = rel.getRelationName.substring(2)
              labelArgMap.get(relLabel).exists({ buffer =>
                buffer.forall { origRel: Relation =>
                  (origRel.getTarget.getStartSpan < rel.getTarget.getStartSpan) &&
                    (origRel.getTarget.getEndSpan < rel.getTarget.getEndSpan)
                }
              })
            })
            countList += s"CArgsOrder:$cArgsOrder"
          }
        })
      })

    countList.groupBy(i => i).mapValues(_.size).foreach({
      case (key, count) =>
        logger.info(s"$key // $count")
    })
    logger.info(countList.groupBy(i => i).mapValues(_.size).toString())
    logger.info(s"Current View: $evaluationViewName")
    logger.info(s"Available Views: $availableViews")

    val documentsWithoutView = annotatedDataset._2.size
    logger.info(s"Documents without predicted view. $documentsWithoutView")
    closeDatabase()
  }

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
      case ex: Exception =>
        logger.error("Error while reading cache file", ex)
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
