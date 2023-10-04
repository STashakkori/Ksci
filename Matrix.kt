// $t@$h     QVLx Labs

package kotlab

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.inverse.InvertMatrix
import org.nd4j.linalg.ops.transforms.Transforms

class Matrix {
  fun add(matrix1: INDArray, matrix2: INDArray): INDArray {
    require(matrix1.shape().contentEquals(matrix2.shape()))
      { "Add: Matrix dimensions must match." }
    return matrix1.add(matrix2)
  }

  fun subtract(matrix1: INDArray, matrix2: INDArray): INDArray {
    require(matrix1.shape().contentEquals(matrix2.shape()))
      { "Subtract: Matrices dimensions must match." }
    return matrix1.sub(matrix2)
  }

  fun multiply(matrix1: INDArray, matrix2: INDArray): INDArray {
    require(matrix1.size(1) == matrix2.size(0))
      { "Multiply: Matrices dimensions must be compatible." }
    return matrix1.mmul(matrix2)
  }

  fun transpose(matrix: INDArray): INDArray {
    return matrix.transpose()
  }

  fun scalarMultiply(matrix: INDArray, scalar: Double): INDArray {
    return matrix.mul(scalar)
  }

  fun inverse(matrix: INDArray): INDArray {
    require(matrix.rows() == matrix.columns())
      { "Inverse: Matrix must be square." }
    return InvertMatrix.invert(matrix, false)
  }

  fun determinant(matrix: INDArray): Double {
    require(matrix.rows() == matrix.columns())
      { "Determinant: Matrix must be square." }
    return Transforms.log(Nd4j.diag(matrix)).sumNumber().toDouble()
  }

  fun identity(matrix: INDArray): INDArray {
    return Nd4j.eye(matrix.rows().toLong())
  }

  fun zeros(rows: Int, cols: Int): INDArray {
    return Nd4j.zeros(rows, cols)
  }

  fun ones(rows: Int, cols: Int): INDArray {
    return Nd4j.ones(rows, cols)
  }

  fun random(rows: Int, cols: Int): INDArray {
    return Nd4j.rand(rows, cols)
  }

  fun create(data: Array<DoubleArray>): INDArray {
    val matrixData = Array(data.size) { DoubleArray(data[0].size) }
    for (i in 0 until data.size) { matrixData[i] = data[i] }
    return Nd4j.create(matrixData)
  }

  fun covariance(data: INDArray): INDArray {
    val centered = data.sub(data.mean(0))
    return (centered.transpose().mmul(centered)).div(data.rows().toDouble())
  }

  fun elementsAdd(matrix1: INDArray, matrix2: INDArray): INDArray {
    require(matrix1.shape().contentEquals(matrix2.shape()))
      { "elementsAdd: Matrices dimensions must match." }
    return matrix1.addi(matrix2)
  }

  fun elementsSubtract(matrix1: INDArray, matrix2: INDArray): INDArray {
    require(matrix1.shape().contentEquals(matrix2.shape()))
      { "elementsSubtract: Matrices dimensions must match." }
    return matrix1.subi(matrix2)
  }

  fun elementsMultiply(matrix1: INDArray, matrix2: INDArray): INDArray {
    require(matrix1.shape().contentEquals(matrix2.shape()))
      { "elementsMultiply: Matrix dimensions must match." }
    return matrix1.muli(matrix2)
  }

  fun elementsDivide(matrix1: INDArray, matrix2: INDArray): INDArray {
    require(matrix1.shape().contentEquals(matrix2.shape()))
      { "elementsDivide: Matrix dimensions must match." }
    return matrix1.divi(matrix2)
  }
}
