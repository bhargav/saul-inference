## Learners
Learners get an input and make a prediction. It can be a classifier, a regression (which are both trained) or any rule-based system. 
Here are the basic parameters used definition of a learners:  

  - `label`: A `Property` which defines the output prediction. In case of classification 
    problems, it is the "category" of the one object. For example, in a document classification 
    task, the category of one text document can be related to its topic, e.g. Sport, politics, etc.
  - `feature`: A set of `Property`s fed to the learner as the input. For example in the case of classification, the classifiers will be trained based on such properties. Example properties are set of words that occur in a document (Bag of words).
  - `classifier`: defines the type of the learning algorithm used. 

Here is an example classifier:

```scala
object OrgClassifier extends Learnable[ConllRawToken](ErDataModelExample) {
  def label: Property[ConllRawToken] = entityType is "Org"
  override def feature = using(word, phrase, containsSubPhraseMent, containsSubPhraseIng,
    containsInPersonList, wordLen, containsInCityList)
  override lazy val classifier = new SparseNetworkLearner()
}
```
### Training and Testing classifiers

-Call `train()` method to train your classifier using the populated data in the data model's training instances:

```scala
OrgClassifier.learn(numberOfIterations)
```
where number of iteration determines how many times the training algorithm should iterate over training data.

-Call `test()` method to test your classifier using the populated data model's test instance:

```scala
OrgClassifier.test()
```

### Availale algorithms 
Here is a list of available algorithms in Saul:
 - [LBJava learning algorithms](https://github.com/IllinoisCogComp/lbjava/blob/master/lbjava/doc/ALGORITHMS.md)
 - [Weka learning algorithms](saul-core/src/main/java/edu/illinois/cs/cogcomp/saul/learn/SaulWekaWrapper.md)


### Saving and loading classifiers
 Simply call the `save()` method:

```scala
OrgClassifier.save()
```

By default the classifier will be save into two files (a `.lc` model file and a `.lex` lexicon file). In order to
 save the classifier in another location, you can set the location in parameter `modelDir`; for example:

```scala
OrgClassifier.modelDir = "myFancyModels/"
OrgClassifier.save()
```

This will save the two model files into the directory `myFancyModels`.

To load the models you can call the `load()` method.

```scala
OrgClassifier.load()
```

If you have different versions of the same classifier (say, different features, different number of iterations, etc),
you can add a suffix to the model files of each variation:

```scala
OrgClassifier.modelSuffix = "20-iterations"
OrgClassifier.save()
```

This would add the suffix "20-iterations" to the files of the classifier at the time of saving them. Note that at
the time of calling `load()` method it will look for model files with suffix "20-iterations".

## Constrained Classifiers
A constrained classifiers is a classifier that predicts the class labels subject to a specified constraints.
Here is the general form: 


```scala
object CONSTRAINED_CLASSIFIER extends ConstraintClassifier[INPUT_TYPE, HEAD_TYPE] {
  override lazy val onClassifier = CLASSIFIER
  override def subjectTo = Some(CONSTRAINT) // optional 
  override def pathToHead = Some(PATH-TO-HEAD-EDGE) // optional 
  override def filter: (t: INPUT_TYPE, h:HEAD_TYPE) => Boolean // optional 
  override def solverType = SOLVER // optional  
}
```

Here we describe each of the parameters in the above snippet: 

 - `CONSTRAINED_CLASSIFIER`: the name of your desired constrained classifier 
 - `INPUT_TYPE`: the input type of the desired constrained classifier 
 - `HEAD_TYPE`: the inference starts from the head object. This type often subsumes `INPUT_TYPE`. For example if we define 
    constraints over sentences while making predictions for each word, `INPUT_TYPE` would be a consituent type in the 
    sentence, while `HEAD_TYPE` would be a sentential type. 
 - `CLASSIFIER`: The base classifier based on which the confidence scores of the constrained inference problem is set. 
 - `CONSTRAINT`: The constraint definition. For more details see the section below. 
 - `SOLVER`: The ILP solver machine used for the inference. Here are the possible values for `solverType`:
    - `OJAlgo`: The [OjAlgo solver](http://ojalgo.org/), an opensource solver.  
    - `Gurobi`: Gurobi, a powerful industrial solver. 
    - `Balas`: Egon Balas' zero-one ILP solving algorithm. It is a branch and bound algorithm that can return the best 
    solution found so far if stopped early.  
 More details can be found in the [`illinois-inference` package](https://gitlab-beta.engr.illinois.edu/cogcomp/inference/). 
 - `PATH-TO-HEAD-EDGE`:  Returns only one object of type `HEAD_TYPE` given an instance of `INPUT_TYPE`; if there are many 
            of them i.e. `Iterable[HEAD]` then it simply returns the head object.
 - `filter`: The function is used to filter the generated candidates from the head object; remember that 
         the inference starts from the head object. This function finds the objects of type `INPUT_TYPE` which are 
         connected to the target object of type `HEAD_TYPE`. If we don't define `filter`, by default it returns all 
         objects connected to `HEAD_TYPE`. The filter is useful for the `JointTraining` when we go over all 
         global objects and generate all contained objects that serve as examples for the basic classifiers involved in 
         the `JoinTraining`. It is possible that we do not want to use all possible candidates but some of them, for 
         example when we have a way to filter the negative candidates, this can come in the filter.

Here is an example usage of this definition: 

```scala
object OrgConstrainedClassifier extends ConstrainedClassifier[ConllRawToken, ConllRelation] {
    override lazy val onClassifier = EntityRelationClassifiers.OrganizationClassifier
    override def pathToHead = Some(-EntityRelationDataModel.pairTo2ndArg)
    override def subjectTo = Some(EntityRelationConstraints.relationArgumentConstraints)
    override def filter(t: ConllRawToken, h: ConllRelation): Boolean = t.wordId == h.wordId2
    override def solverType = OJAlgo
}
```

In this example, the base (non-constrained) classifier is `OrganizationClassifier` which predicts whether the given instance 
(of type `ConllRawToken`) is an organization or not. Since the constraints `relationArgumentConstraints` are defined over 
triples (i.e two entities and the relation between them), the head type is defined as `ConllRelation` (which is 
relatively more general than `ConllRawToken`). The filter function ensures that the head relation corresponds to the given 
input entity token. 

**Tip:** The constrained classifier is using in-memory caching to make the inference faster. If you want to turn off caching 
    just include `override def useCaching = false` in body of the classifier definition. 

### Constraints
A "constraint" is a logical restriction over possible values that can be assigned to a number of variables;
For example, a binary constraint could be `{if {A} then NOT {B}}`.
In Saul, the constraints are defined for the assignments to class labels. In what follows we outine the details of operators 
which help us define the constraints. Before jumping into the details, note that you have to have the following import 
in order to have the following operators work: 

```scala 
import edu.illinois.cs.cogcomp.saul.infer.Constraint._
```

#### Propositional constraints
  This defines constraint on the prediction of a classifier on a given instance. Here is the basic form. Consider an 
   imaginary classifier `SomeClassifier` which returns `A` or  `B`. Here is how we create propositional constraint 
   to force the prediction on instance `x` to have label `A`: 
```  
  SomeClassifier on x is "A"
```  

In the above definition, `on` and `is` are keywords. 

Here different variations and extensions to this basic usage: 

 - If the label were `true` and `false`, one can use `isTrue` instead of `is "true"` (and similarily `isFalse` instead of `is "false"`). 
 - If instead of equality you want to use inequality, you can use the keyword `isNot`, instead of `is`. 
 - If you want to use the equality on multiple label values, you can use the `isOneOf(.)` keywors, instead of `is`. 
 - If you want a classifier have the same label on two different instances, you can do: 

```scala 
  SomeClassifier on x1 equalsTo x2 
```
   Similarly if you want a classifier have the different label on two different instances, you can do:

```scala 
  SomeClassifier on x1 differentFrom x2 
```

 - If you want two different classifiers have the same labels on a instances, you can do:

```scala 
  SomeClassifier1 on x equalsTo SomeClassifier2 
```

   And similarly if you want two different classifiers have different labels on a instances, you can do:

```scala 
  SomeClassifier1 on x differentFrom SomeClassifier2 
```


#### Binary and Unary binary operators 

One way of combining base logical rules is applying binary operations (for example conjunction, etc). 

| Operator |  Name | Definition |  Example    |
|----------|---------------|---------|------|
| `and`    |  conjunction  | A binary operator to create a conjunction of the two constraints before and after it  |  `(SomeClassifier1 on x is "A1") and  (SomeClassifier2 on y is "A2")`  |
| `or`     |  disjunction  | A binary operator to create a disjunction of the two constraints before and after it  |  `(SomeClassifier1 on x is "A1") or  (SomeClassifier2 on y is "A2")`    |
|  `==>`   |  implication  | The implication operator, meaning that if the contraint before it is true, the constraint following it must be true as well  |  `(SomeClassifier1 on x is "A1") ==> (SomeClassifier2 on y is "A2")`  |
|  `<==>`   |  double implication  | The double-implication operator (aka "if and only if"), meaning that if the contraint before it is true, if and only if the constraint following it is true as well  |  `(SomeClassifier1 on x is "A1") <==> (SomeClassifier2 on y is "A2")`  |
|  `!`     |  negation     | A prefix unary operator to negate the effect of the constraint following it.  |   `!(SomeClassifier1 on x is "A1")`   |

#### Collection operators 

This operators distribute the definitions of the constraints over collections. Here are the definition and examples: 

| Operator | Definition |  Example  |
|----------|------------|---------|---|
| `ForEach`  |  This operator works only on `Node`s. For each single instance in the node. This is often times one of the starting points for defining constraints. So if you are defining using a constrained classifier with head type `HEAD_TYPE`, we the definition of the constraint have to start with the node corresponding to this type.  |  `textAnnotationNode.ForEach { x: TextAnnotation => Some-Constraint-On-X }`   |     
| `ForAll`   |  For **all** the elements in the collection it applies the constraints. In other words, the constrain should hold for **all** elements of the collection.   |  `textAnnotationNode.ForAll { x: TextAnnotation => Some-Constraint-On-x }`  |    
| `Exists`    | The constrain should hold for **at least one** element of the collection.   |  `textAnnotationNode.Exists { x: TextAnnotation => Some-Constraint-On-x }` | 
| `AtLeast(k: Int)`  |  The constrain should hold for **at least `k`** elements of the collection.  |  `textAnnotationNode.AtLeast(2) { x: TextAnnotation => Some-Constraint-On-x }` |  
| `AtMost(k: Int)`  |  The constrain should hold for **at most `k`** elements of the collection.  | `textAnnotationNode.AtMost(3) { x: TextAnnotation => Some-Constraint-On-x }`  | 
| `Exactly(k: Int)`  | The constrain should hold for **exactly `k`** elements of the collection.  |  `textAnnotationNode.Exactly(3){ x: TextAnnotation => Some-Constraint-On-x }`  | 

**Tip:** Except `ForEach` which is used only on nodes, all the above operators can be used as postfix operator on the list of constraints. For example: 

```scala 
val constraintCollection = for { 
   // some complicated loop variables 
}
 yield someConstraint

constraintCollection.ForAll 
```


There are just the definitions of the operations. If you want to see real examples of the operators in actions see [the definitions of constraints for ER-example](https://github.com/IllinoisCogComp/saul/blob/master/saul-examples/src/main/scala/edu/illinois/cs/cogcomp/saulexamples/nlp/EntityRelation/EntityRelationConstraints.scala). 

**Tip:** Note whenever the constrained inference is infeasible (i.e. the constraints are overly tight), we use the default 
prediction of the base classifier. Hence if you see the performance of the constrained classifier is very close to the performance 
of the base classifier it's probably most of your inference problems are becoming infeasible. In such cases it is worth verifying 
the correctness of your constraint definitions. 

#### Common mistakes in using constrained classifiers 
 - Not defining the constraints with `def` keyword (instead defining them with the `val` keyword). The `def` keyword 
    makes the propositionalization of the constraints lazy, i.e. it waits until you call them and then they get 
    evaluated. 
 - If you face the following error: 
 ```
 requirement failed: The target value Some(SomeLabel) is not a valid value for classifier ClassifierName with the tag-set: Set(SomeTags)
 ```
 it often means that it is constraiend to have a label which it does not contain in output label lexicon. Another reason 
 for this can be not loading the base classifier model properly. 
 