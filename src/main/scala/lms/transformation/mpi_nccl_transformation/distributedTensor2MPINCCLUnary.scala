package lms.transformation.tensor

import scala.annotation.implicitNotFound
import scala.collection._

import lms.core._
import lms.core.stub._
import lms.collection.mutable._
import lms.macros.SourceContext
import lms.thirdparty.{RandomDataTypeLess, NCCLTypeLess, MPIOps, NCCLOps, SIZE_TTypeLess}
import lms.thirdparty.array_computation.{ArrayCPUTypeLess, CUDATypeLess, CUBLASTypeLess, CudaOps}
import lms.transformation.util.DataStructure

import Backend._

trait DistributedTensor2MPI_NCCLUnary extends DistributeTensor2MPI_NCCLBase {

  import BaseTypeLess._
  import PrimitiveTypeLess._
  import RangeTypeLess._
  import ArrayTypeLess._
  import ArrayCPUTypeLess._
  import FixedSizeDistributedTensorTypeLess._
  import CUDATypeLess._
  import RandomDataTypeLess._
  import NCCLTypeLess._
  import SIZE_TTypeLess._
  import CUBLASTypeLess._

  // helper function for computing element-wise negation in GPUs
  val CUDA_NEGATE_KERNEL_MAP = scala.collection.mutable.HashMap[Manifest[_], (TOP, TOP, TOP, DIM3, DIM3) => UNIT]()
  def CUDA_NEGATE_FUN(m: Manifest[_])(implicit __pos: SourceContext) = CUDA_NEGATE_KERNEL_MAP.getOrElseUpdate(m, CUDA_NEGATE_KERNEL(m))
  def gpu_neg_array(size: Int, m: Manifest[_], device: INT, operand: Backend.Exp)(implicit __pos: SourceContext): ARRAY =
	withComment(s"computing NEG on GPU for size $size and type $m at device (pre-rename) ${device.x} with operand $operand") {
		val array = gpu_array(size, m, device)
		val neg_fun = CUDA_NEGATE_FUN(m)
		neg_fun(new ARRAY(operand), array, size, DIM3(gridSize), DIM3(blockSize))
		array
  }

  val CUDA_INVERT_KERNEL_MAP = scala.collection.mutable.HashMap[Manifest[_], (TOP, TOP, TOP, DIM3, DIM3) => UNIT]()
  def CUDA_INVERT_FUN(m: Manifest[_])(implicit __pos: SourceContext) = CUDA_INVERT_KERNEL_MAP.getOrElseUpdate(m, CUDA_INVERT_KERNEL(m))
  def gpu_inv_array(size: Int, m: Manifest[_], device: INT, operand: Backend.Exp)(implicit __pos: SourceContext): ARRAY =
	withComment(s"computing INV on GPU for size $size and type $m at device (pre-rename) ${device.x} with operand $operand") {
		val array = gpu_array(size, m, device)
		val inv_fun = CUDA_INVERT_FUN(m)
		inv_fun(new ARRAY(operand), array, size, DIM3(gridSize), DIM3(blockSize))
		array
  }

  val CUDA_TANH_KERNEL_MAP = scala.collection.mutable.HashMap[Manifest[_], (TOP, TOP, TOP, DIM3, DIM3) => UNIT]()
  def CUDA_TANH_FUN(m: Manifest[_])(implicit __pos: SourceContext) = CUDA_TANH_KERNEL_MAP.getOrElseUpdate(m, CUDA_TANH_KERNEL(m))
  def gpu_tanh_array(size: Int, m: Manifest[_], device: INT, operand: Backend.Exp)(implicit __pos: SourceContext): ARRAY =
	withComment(s"computing TANH on GPU for size $size and type $m at device (pre-rename) ${device.x} with operand $operand") {
		val array = gpu_array(size, m, device)
		val tanh_fun = CUDA_TANH_FUN(m)
		tanh_fun(new ARRAY(operand), array, size, DIM3(gridSize), DIM3(blockSize))
		array
  }

  val CUDA_RELU_KERNEL_MAP = scala.collection.mutable.HashMap[Manifest[_], (TOP, TOP, TOP, DIM3, DIM3) => UNIT]()
  def CUDA_RELU_FUN(m: Manifest[_])(implicit __pos: SourceContext) = CUDA_RELU_KERNEL_MAP.getOrElseUpdate(m, CUDA_RELU_KERNEL(m))
  def gpu_relu_array(size: Int, m: Manifest[_], device: INT, operand: Backend.Exp)(implicit __pos: SourceContext): ARRAY =
	withComment(s"computing RELU on GPU for size $size and type $m at device (pre-rename) ${device.x} with operand $operand") {
		val array = gpu_array(size, m, device)
		val relu_fun = CUDA_RELU_FUN(m)
		relu_fun(new ARRAY(operand), array, size, DIM3(gridSize), DIM3(blockSize))
		array
  }

  val CUDA_TRANSPOSE_KERNEL_MAP = scala.collection.mutable.HashMap[Manifest[_], (TOP, TOP, TOP, DIM3, DIM3) => UNIT]()
  def CUDA_TRANSPOSE_FUN(m: Manifest[_])(implicit __pos: SourceContext) = CUDA_TRANSPOSE_KERNEL_MAP.getOrElseUpdate(m, CUDA_TRANSPOSE_KERNEL(m))
  def gpu_transpose_array(size: Int, m: Manifest[_], device: INT, operand: Backend.Exp)(implicit __pos: SourceContext): ARRAY =
	withComment(s"computing TRANSPOSE on GPU for size $size and type $m at device (pre-rename) ${device.x} with operand $operand") {
		val array = gpu_array(size, m, device)
		val transpose_fun = CUDA_TRANSPOSE_FUN(m)
		transpose_fun(new ARRAY(operand), array, size, DIM3(gridSize), DIM3(blockSize))
		array
  }

  // val unaryOps = List("tensor_negate", "tensor_invert", "tensor_tanh", "tensor_relu", "tensor_transpose")
  val unaryOps = List("tensor_negate", "tensor_invert", "tensor_tanh", "tensor_relu")

  override def transform(n: Node): Backend.Exp = n match {

    case Node(s, op, Backend.Const(tt:TensorType)::Backend.Const(anno:Anno)::(operand:Backend.Sym)::_, _)
      if unaryOps.contains(op) =>

      val sourceTensor = new TENSOR(s, useOldMetadata = true)
      implicit val pos: SourceContext = sourceTensor.pos
      val m = sourceTensor.et
      val count = numeral(tt.shapeSize)

      op match {
        case "tensor_negate" => gpu_neg_array(count, m, myCUDADevice, transform(operand)).x
        case "tensor_invert" => gpu_inv_array(count, m, myCUDADevice, transform(operand)).x
        case "tensor_tanh" => gpu_tanh_array(count, m, myCUDADevice, transform(operand)).x
        case "tensor_relu" => gpu_relu_array(count, m, myCUDADevice, transform(operand)).x
        case "tensor_transpose" => gpu_transpose_array(count, m, myCUDADevice, transform(operand)).x
        case _ => throw new Exception(s"op $op is not unary op")
      }

    case _ => super.transform(n)
  }
}
