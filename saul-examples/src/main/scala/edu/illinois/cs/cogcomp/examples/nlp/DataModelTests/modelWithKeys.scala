package edu.illinois.cs.cogcomp.examples.nlp.DataModelTests

import edu.illinois.cs.cogcomp.core.datastructures.textannotation._
import edu.illinois.cs.cogcomp.lfs.data_model.DataModel
import edu.illinois.cs.cogcomp.lfs.data_model.DataModel._
import edu.illinois.cs.cogcomp.lfs.data_model.edge.Edge

import scala.collection.mutable.{Map => MutableMap}

/**
 * Created by Parisa on 10/4/15.
 */

object modelWithKeys extends DataModel {

  /** Node Types
    */
  val document = node[TextAnnotation](
    PrimaryKey = {
      t: TextAnnotation => t.getId
    }
  )

  val sentence = node[Sentence](
    PrimaryKey = {
      t: Sentence => t.hashCode().toString
    }
    ,
    SecondaryKeyMap = MutableMap(
      'dTos -> ((t: Sentence) => t.getSentenceConstituent.getTextAnnotation.getId)
    )
  )
  /** Property Types
    */

  val label = discreteAttributeOf[Constituent]('label) {
    x => {
      x.getLabel
    }
  }

  val docFeatureExample = discreteAttributeOf[TextAnnotation]('doc) {
    x: TextAnnotation => {
      x.getNumberOfSentences.toString
    }
  }
  val sentenceFeatureExample = discreteAttributeOf[Sentence]('sentence) {
    x: Sentence => {
      x.getText
    }
  }

  /** Edge Types
    */

  val docTosen = edge[TextAnnotation, Sentence]('dTos)

  val NODES = List(document, sentence)
  val PROPERTIES = List(docFeatureExample, sentenceFeatureExample)
  val EDGES: List[Edge[_, _]] = docTosen

}
