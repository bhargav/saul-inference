package edu.illinois.cs.cogcomp.saulexamples.nlp.DataModelTests

import edu.illinois.cs.cogcomp.saul.datamodel.DataModel

/** @author sameer
  * @since 10/8/15.
  */
object GraphQueries extends App {

  val model = new DataModel {
    val token = node[String]
    val sentence = node[String]
    val document = node[String]
    val corpus = node[String]

    val t2s = edge(token, sentence, 't2s)
    val s2d = edge(sentence, document, 's2d)
    val d2c = edge(document, corpus, 'd2c)
  }

  import model._

  println(token ~> t2s == sentence)
  println(token ~> t2s <~ t2s == token)
  println("---")
  println(token ~> t2s ~> s2d == document)
  println(token ~> t2s ~> s2d <~ s2d == sentence)
  println(token ~> t2s ~> s2d <~ s2d <~ t2s == token)
  println("---")
  println(token ~> t2s ~> s2d ~> d2c == corpus)
  println(token ~> t2s ~> s2d ~> d2c <~ d2c <~ s2d <~ t2s == token)
}
