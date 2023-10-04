// $t@$h     QVLx Labs

package kotlab

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.inverse.InvertMatrix
import org.nd4j.linalg.ops.transforms.Transforms

class MatrixUtils {
  fun add(matrix1: INDArray, matrix2: INDArray): INDArray {
    require(matrix1.shape().contentEquals(matrix2.shape()))
      { "Matrix dimensions must match for addition." }
    return matrix1.add(matrix2)
  }

  fun subtract(matrix1: INDArray, matrix2: INDArray): INDArray {
    require(matrix1.shape().contentEquals(matrix2.shape()))
      { "Matrix dimensions must match for subtraction." }
    return matrix1.sub(matrix2)
  }

  fun multiply(matrix1: INDArray, matrix2: INDArray): INDArray {
    require(matrix1.size(1) == matrix2.size(0))
      { "Matrix dimensions are incompatible for multiplication." }
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
      { "Matrix must be square for inversion." }
    return InvertMatrix.invert(matrix, false)
  }

  fun determinant(matrix: INDArray): Double {
    require(matrix.rows() == matrix.columns())
      { "Matrix must be square for determinant calculation." }
    return Transforms.log(Nd4j.diag(matrix)).sumNumber().toDouble()
  }

  fun identityMatrix(matrix: INDArray): INDArray {
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

  fun createMatrix(data: Array<DoubleArray>): INDArray {
    val matrixData = Array(data.size) { DoubleArray(data[0].size) }
    for (i in 0 until data.size) { matrixData[i] = data[i] }
    return Nd4j.create(matrixData)
  }

  fun covarianceMatrix(data: INDArray): INDArray {
    val centered = data.sub(data.mean(0))
    return (centered.transpose().mmul(centered)).div(data.rows().toDouble())
  }
}
