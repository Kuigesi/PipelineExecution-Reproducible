package lms.transformation.tensor

import scala.annotation.implicitNotFound
import scala.collection._

import lms.core._
import lms.core.stub._
import lms.collection.mutable._
import lms.macros.SourceContext
import lms.thirdparty.array_computation.{ArrayCPUOps, CUDATypeLess, CudaOps}

import Backend._

trait FixedSizeDistributedTensorSplitConcatTypeLess extends FixedSizeDistributedTensorMutationTypeLess {
  import BaseTypeLess._

  def Split(x: TENSOR, axis: Int, slices: List[Int], anno: Anno = NAnno)(implicit __pos: SourceContext): TENSORS = {
    val x_resultType = x.resultType
    val x_tensorShape = x_resultType.shape
    assert(slices.sum == x_tensorShape(axis).size)
    val resultTensorShapes = slices map { s =>
      // FIXME(feiw) we asked the split result to use the same dim at the split axis. is this correct?
      x_tensorShape.zipWithIndex.map {
        case(a, i) => if (i == axis) Size(a.dim, s) else a
      }
    }
    val resultTypes = resultTensorShapes map { s =>
      TensorType(s, x_resultType.et)
    }
    new TENSORS(Adapter.g.reflectRead("tensors_split", C(resultTypes), C(anno), x.x, C(axis))(x.x)).withSrcType(__pos, x.et)
  }

  def Concat(xs: List[TENSOR], axis: Int, anno: Anno = NAnno)(implicit __pos: SourceContext): TENSOR = {
    // FIXME(feiw) assert shape and element type
    // need to check that the inputs have the same shape except at the `axis` dimension.
    val elementTypes = xs.map(_.resultType.et)
    require(elementTypes.length > 0, "there must be at least one TENSOR to concat")
    require(elementTypes.forall(_ == elementTypes.head), "all TENSORs must have the same type")
    val tensorShapes = xs.map(_.resultType.shapeSize)
    def sameShapeExceptDim(left: Seq[Int], right: Seq[Int], axis: Int) =
      left.zip(right).zipWithIndex.forall { case ((l, r), i) => i == axis || l == r }
    require(tensorShapes.forall(sameShapeExceptDim(tensorShapes.head, _, axis)),
      s"all TENSORs must have the same shape except dim $axis")

    val concatSize = xs.map(_.resultType).map(_.shape).map(s=>s(axis)).map(_.size).sum
    val concatShape = xs(0).resultType.shape.zipWithIndex.map {
      case(s, i) => if (i == axis) Size(s.dim, concatSize) else s
    }
    val concatType = TensorType(concatShape, xs(0).et)
    val defs = xs.map(_.x).toSeq
    val all_defs = (C(concatType)::C(anno)::C(axis)::xs.map(_.x)).toSeq
    (new TENSOR(Adapter.g.reflectRead("tensor_concat", all_defs:_*)(defs:_*))).withSrcType(__pos, xs(0).et)
  }

  override def mergable_dims(node: Node) = node match {
    case Node(s, "tensors_split", _, _) => List()
    case Node(s, "tensor_concat", tt::anno::Backend.Const(axis:Int)::(inputs:List[Backend.Sym])::_, _) =>
      val input_types: List[Seq[Dim]] = inputs.map(x => (new TENSOR(x, useOldMetadata=true)).resultType.shapeDim)
      input_types.transpose.zipWithIndex.flatMap { case (dims: List[Dim], index) =>
        if (index != axis) dims.init zip dims.tail else List()
      }
    case _ => super.mergable_dims(node)
  }

  override def aircopCollect(node: Node, forwardNodes: mutable.ArrayBuffer[Node],
      weightNodes: mutable.ArrayBuffer[Node], backwardNodes: mutable.ArrayBuffer[()=>Unit], liftNodes: mutable.Set[Backend.Sym],
      gradMap: GradMapWrapper,
      momentumMap: mutable.HashMap[Backend.Sym, TENSOR],
      transform: Backend.Exp => Backend.Exp) = node match {

    case Node(s, "tensors_split", tts::Backend.Const(anno:Anno)::(x:Backend.Sym)::Backend.Const(axis:Int)::_, _) =>
      implicit val pos: SourceContext = Adapter.oldSourceMap(s)
      // save forward op in forwardNodes
      forwardNodes += node
      // save backward op in backwardNodes
      (() => {
        val splitOp = new TENSORS(s, useOldMetadata=true)
        val grads = gradMap.getGradsOfOp(s)
        val c_grads = Concat(grads, axis, anno)
        Accumulate(gradMap(x), c_grads, anno); ()
      }) +=: backwardNodes

    case _ =>
      super.aircopCollect(node, forwardNodes, weightNodes, backwardNodes, liftNodes, gradMap, momentumMap, transform)
  }

  override def printTensor(node: Node, graph: Graph): String = node match {
    case Node(s, "tensors_split", Backend.Const(tts:List[TensorType])::Backend.Const(anno:Anno)::(x:Backend.Sym)::Backend.Const(axis:Int)::_, _) =>
      s"$s:${tts.length} = tensors_split($x, axis=$axis) (${symTensorShape(x, graph)})->(${tts.map(_.toString).mkString(", ")})${if (anno != NAnno) s"\nAnno: $anno" else ""}"
    case Node(s, "tensor_concat", Backend.Const(tt:TensorType)::Backend.Const(anno:Anno)::Backend.Const(axis:Int)::(inputs:List[Backend.Sym])::_, _) =>
      s"$s = tensor_concat(${inputs.mkString(",")}) (${inputs.map(symTensorShape(_, graph)).mkString(", ")})->${tt.toString}${if (anno != NAnno) s"\nAnno: $anno" else ""}"
    case _ => super.printTensor(node, graph)
  }
}


trait FixedSizeDistributedTensorOpsSplitConcat extends FixedSizeDistributedTensorOpsBase {
  import FixedSizeDistributedTensorTypeLess._

  implicit class TensorOpsSplit[T:Numeric:Manifest](x: Rep[Tensor[T]]) {
    val self = tensor(x)

    def split(axis: Int, slices: List[Int])(implicit __pos: SourceContext, anno: Anno): List[Rep[Tensor[T]]] = {
      val op = Split(self, axis, slices, anno)
      ((0 until slices.length): Range).toList.map(i => Wrap[Tensor[T]](TENSORS.getResult(op, i).x))
    }
    def split(axis: Int, slices: List[Int], anno: Anno)(implicit __pos: SourceContext): List[Rep[Tensor[T]]] = {
      val op = Split(self, axis, slices, anno)
      ((0 until slices.length): Range).toList.map(i => Wrap[Tensor[T]](TENSORS.getResult(op, i).x))
    }
  }
}
