/** This software is released under the University of Illinois/Research and Academic Use License. See
  * the LICENSE file in the root folder for details. Copyright (c) 2016
  *
  * Developed by: The Cognitive Computations Group, University of Illinois at Urbana-Champaign
  * http://cogcomp.cs.illinois.edu/
  */
package edu.illinois.cs.cogcomp.saul.classifier.infer.factorgraph

import cc.factorie.la.{ DenseTensor2, DenseTensor3 }
import cc.factorie.model.{ ConstantWeights2, ConstantWeights3, DotFamilyWithStatistics2, DotFamilyWithStatistics3, Weights2, Weights3 }

private[factorgraph] object PairConjunctionUtils {
  val family2PP = new DotFamilyWithStatistics2[BinaryRandomVariable, BinaryRandomVariable] {

    private val weightsValue = new DenseTensor2(2, 2)
    weightsValue.update(0, 0, Double.NegativeInfinity)
    weightsValue.update(0, 1, Double.NegativeInfinity)
    weightsValue.update(1, 0, Double.NegativeInfinity)
    weightsValue.update(1, 1, 1.0)

    override def weights: Weights2 = new ConstantWeights2(weightsValue)

    override type FamilyType = this.type
  }

  val family2PN = new DotFamilyWithStatistics2[BinaryRandomVariable, BinaryRandomVariable] {

    private val weightsValue = new DenseTensor2(2, 2)
    weightsValue.update(0, 0, Double.NegativeInfinity)
    weightsValue.update(0, 1, Double.NegativeInfinity)
    weightsValue.update(1, 0, 1.0)
    weightsValue.update(1, 1, Double.NegativeInfinity)

    override def weights: Weights2 = new ConstantWeights2(weightsValue)

    override type FamilyType = this.type
  }

  val family2NP = new DotFamilyWithStatistics2[BinaryRandomVariable, BinaryRandomVariable] {

    private val weightsValue = new DenseTensor2(2, 2)
    weightsValue.update(0, 0, Double.NegativeInfinity)
    weightsValue.update(0, 1, 1.0)
    weightsValue.update(1, 0, Double.NegativeInfinity)
    weightsValue.update(1, 1, Double.NegativeInfinity)

    override def weights: Weights2 = new ConstantWeights2(weightsValue)

    override type FamilyType = this.type
  }

  val family2NN = new DotFamilyWithStatistics2[BinaryRandomVariable, BinaryRandomVariable] {

    private val weightsValue = new DenseTensor2(2, 2)
    weightsValue.update(0, 0, 1.0)
    weightsValue.update(0, 1, Double.NegativeInfinity)
    weightsValue.update(1, 0, Double.NegativeInfinity)
    weightsValue.update(1, 1, Double.NegativeInfinity)

    override def weights: Weights2 = new ConstantWeights2(weightsValue)

    override type FamilyType = this.type
  }

  /* Note: Third variable is the output variable, whose assignment is forced to agree with the conjunction */
  val family2WithOutputPPP = new DotFamilyWithStatistics3[BinaryRandomVariable, BinaryRandomVariable, BinaryRandomVariable] {

    private val weightsValue = new DenseTensor3(2, 2, 2)
    weightsValue.update(0, 0, 0, 1.0)
    weightsValue.update(0, 1, 0, 1.0)
    weightsValue.update(1, 0, 0, 1.0)
    weightsValue.update(1, 1, 0, Double.NegativeInfinity)
    weightsValue.update(0, 0, 1, Double.NegativeInfinity)
    weightsValue.update(0, 1, 1, Double.NegativeInfinity)
    weightsValue.update(1, 0, 1, Double.NegativeInfinity)
    weightsValue.update(1, 1, 1, 1.0)

    override def weights: Weights3 = new ConstantWeights3(weightsValue)

    override type FamilyType = this.type
  }

  /* Note: Third variable is the output variable, whose assignment is forced to agree with the conjunction */
  val family2WithOutputPPN = new DotFamilyWithStatistics3[BinaryRandomVariable, BinaryRandomVariable, BinaryRandomVariable] {

    private val weightsValue = new DenseTensor3(2, 2, 2)
    weightsValue.update(0, 0, 0, Double.NegativeInfinity)
    weightsValue.update(0, 1, 0, Double.NegativeInfinity)
    weightsValue.update(1, 0, 0, Double.NegativeInfinity)
    weightsValue.update(1, 1, 0, 1.0)
    weightsValue.update(0, 0, 1, 1.0)
    weightsValue.update(0, 1, 1, 1.0)
    weightsValue.update(1, 0, 1, 1.0)
    weightsValue.update(1, 1, 1, Double.NegativeInfinity)

    override def weights: Weights3 = new ConstantWeights3(weightsValue)

    override type FamilyType = this.type
  }

  /* Note: Third variable is the output variable, whose assignment is forced to agree with the conjunction */
  val family2WithOutputPNP = new DotFamilyWithStatistics3[BinaryRandomVariable, BinaryRandomVariable, BinaryRandomVariable] {

    private val weightsValue = new DenseTensor3(2, 2, 2)
    weightsValue.update(0, 0, 0, 1.0)
    weightsValue.update(0, 1, 0, 1.0)
    weightsValue.update(1, 0, 0, Double.NegativeInfinity)
    weightsValue.update(1, 1, 0, 1.0)
    weightsValue.update(0, 0, 1, Double.NegativeInfinity)
    weightsValue.update(0, 1, 1, Double.NegativeInfinity)
    weightsValue.update(1, 0, 1, 1.0)
    weightsValue.update(1, 1, 1, Double.NegativeInfinity)

    override def weights: Weights3 = new ConstantWeights3(weightsValue)

    override type FamilyType = this.type
  }

  /* Note: Third variable is the output variable, whose assignment is forced to agree with the conjunction */
  val family2WithOutputPNN = new DotFamilyWithStatistics3[BinaryRandomVariable, BinaryRandomVariable, BinaryRandomVariable] {

    private val weightsValue = new DenseTensor3(2, 2, 2)
    weightsValue.update(0, 0, 0, Double.NegativeInfinity)
    weightsValue.update(0, 1, 0, Double.NegativeInfinity)
    weightsValue.update(1, 0, 0, 1.0)
    weightsValue.update(1, 1, 0, Double.NegativeInfinity)
    weightsValue.update(0, 0, 1, 1.0)
    weightsValue.update(0, 1, 1, 1.0)
    weightsValue.update(1, 0, 1, Double.NegativeInfinity)
    weightsValue.update(1, 1, 1, 1.0)

    override def weights: Weights3 = new ConstantWeights3(weightsValue)

    override type FamilyType = this.type
  }

  /* Note: Third variable is the output variable, whose assignment is forced to agree with the conjunction */
  val family2WithOutputNPP = new DotFamilyWithStatistics3[BinaryRandomVariable, BinaryRandomVariable, BinaryRandomVariable] {

    private val weightsValue = new DenseTensor3(2, 2, 2)
    weightsValue.update(0, 0, 0, 1.0)
    weightsValue.update(0, 1, 0, Double.NegativeInfinity)
    weightsValue.update(1, 0, 0, 1.0)
    weightsValue.update(1, 1, 0, 1.0)
    weightsValue.update(0, 0, 1, Double.NegativeInfinity)
    weightsValue.update(0, 1, 1, 1.0)
    weightsValue.update(1, 0, 1, Double.NegativeInfinity)
    weightsValue.update(1, 1, 1, Double.NegativeInfinity)

    override def weights: Weights3 = new ConstantWeights3(weightsValue)

    override type FamilyType = this.type
  }

  /* Note: Third variable is the output variable, whose assignment is forced to agree with the conjunction */
  val family2WithOutputNPN = new DotFamilyWithStatistics3[BinaryRandomVariable, BinaryRandomVariable, BinaryRandomVariable] {

    private val weightsValue = new DenseTensor3(2, 2, 2)
    weightsValue.update(0, 0, 0, Double.NegativeInfinity)
    weightsValue.update(0, 1, 0, 1.0)
    weightsValue.update(1, 0, 0, Double.NegativeInfinity)
    weightsValue.update(1, 1, 0, Double.NegativeInfinity)
    weightsValue.update(0, 0, 1, 1.0)
    weightsValue.update(0, 1, 1, Double.NegativeInfinity)
    weightsValue.update(1, 0, 1, 1.0)
    weightsValue.update(1, 1, 1, 1.0)

    override def weights: Weights3 = new ConstantWeights3(weightsValue)

    override type FamilyType = this.type
  }

  /* Note: Third variable is the output variable, whose assignment is forced to agree with the conjunction */
  val family2WithOutputNNP = new DotFamilyWithStatistics3[BinaryRandomVariable, BinaryRandomVariable, BinaryRandomVariable] {

    private val weightsValue = new DenseTensor3(2, 2, 2)
    weightsValue.update(0, 0, 0, Double.NegativeInfinity)
    weightsValue.update(0, 1, 0, 1.0)
    weightsValue.update(1, 0, 0, 1.0)
    weightsValue.update(1, 1, 0, 1.0)
    weightsValue.update(0, 0, 1, 1.0)
    weightsValue.update(0, 1, 1, Double.NegativeInfinity)
    weightsValue.update(1, 0, 1, Double.NegativeInfinity)
    weightsValue.update(1, 1, 1, Double.NegativeInfinity)

    override def weights: Weights3 = new ConstantWeights3(weightsValue)

    override type FamilyType = this.type
  }

  /* Note: Third variable is the output variable, whose assignment is forced to agree with the conjunction */
  val family2WithOutputNNN = new DotFamilyWithStatistics3[BinaryRandomVariable, BinaryRandomVariable, BinaryRandomVariable] {

    private val weightsValue = new DenseTensor3(2, 2, 2)
    weightsValue.update(0, 0, 0, 1.0)
    weightsValue.update(0, 1, 0, Double.NegativeInfinity)
    weightsValue.update(1, 0, 0, Double.NegativeInfinity)
    weightsValue.update(1, 1, 0, Double.NegativeInfinity)
    weightsValue.update(0, 0, 1, Double.NegativeInfinity)
    weightsValue.update(0, 1, 1, 1.0)
    weightsValue.update(1, 0, 1, 1.0)
    weightsValue.update(1, 1, 1, 1.0)

    override def weights: Weights3 = new ConstantWeights3(weightsValue)

    override type FamilyType = this.type
  }
}
