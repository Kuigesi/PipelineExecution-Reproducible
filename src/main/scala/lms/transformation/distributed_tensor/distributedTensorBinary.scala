package lms.transformation.tensor

import scala.annotation.implicitNotFound
import scala.collection._

import lms.core._
import lms.core.stub._
import lms.collection.mutable._
import lms.macros.SourceContext
import lms.thirdparty.array_computation.{ArrayCPUOps, CUDATypeLess, CudaOps}

import Backend._

trait FixedSizeDistributedTensorBinaryTypeLess extends FixedSizeDistributedTensorMutationTypeLess {

  def ElemWiseNoBroadCasting(x: TENSOR, y: TENSOR, anno: Anno, __pos: SourceContext)(op: String): TENSOR = {
    assert(x.shapeSize == y.shapeSize)
    assert(x.et == y.et)
    (new TENSOR(Adapter.g.reflectRead(op, C(x.resultType), C(anno), x.x, y.x)(x.x, y.x))).withSrcType(__pos, x.et)
  }

  def Add(x: TENSOR, y: TENSOR, anno: Anno = NAnno)(implicit __pos: SourceContext): TENSOR =
    ElemWiseNoBroadCasting(x, y, anno, __pos)("tensor_add")

  def Sub(x: TENSOR, y: TENSOR, anno: Anno = NAnno)(implicit __pos: SourceContext): TENSOR =
    ElemWiseNoBroadCasting(x, y, anno, __pos)("tensor_minus")

  def Mul(x: TENSOR, y: TENSOR, anno: Anno = NAnno)(implicit __pos: SourceContext): TENSOR =
    ElemWiseNoBroadCasting(x, y, anno, __pos)("tensor_mul")

  def Div(x: TENSOR, y: TENSOR, anno: Anno = NAnno)(implicit __pos: SourceContext): TENSOR =
    ElemWiseNoBroadCasting(x, y, anno, __pos)("tensor_div")

  def TanhGrad(x: TENSOR, y: TENSOR, anno: Anno = NAnno)(implicit __pos: SourceContext): TENSOR =
    ElemWiseNoBroadCasting(x, y, anno, __pos)("tensor_tanh_grad")

  def ReluGrad(x: TENSOR, y: TENSOR, anno: Anno = NAnno)(implicit __pos: SourceContext): TENSOR =
    ElemWiseNoBroadCasting(x, y, anno, __pos)("tensor_relu_grad")

  def InvertGrad(x: TENSOR, y: TENSOR, anno: Anno = NAnno)(implicit __pos: SourceContext): TENSOR =
    ElemWiseNoBroadCasting(x, y, anno, __pos)("tensor_invert_grad")

  val binaryOps = List("tensor_add", "tensor_minus", "tensor_mul", "tensor_div", "tensor_tanh_grad", "tensor_relu_grad", "tensor_invert_grad")

  override def mergable_dims(node: Node) = node match {
    case Node(s, op, tt::anno::(x:Backend.Sym)::(y:Backend.Sym)::_, _)
        if binaryOps.contains(op) =>
      val x_type = (new TENSOR(x, useOldMetadata=true)).resultType
      val y_type = (new TENSOR(y, useOldMetadata=true)).resultType
      (x_type.shape zip y_type.shape).toList map { case (a:Size, b:Size) => (a.dim, b.dim) }
    case _ => super.mergable_dims(node)
  }

  override def aircopCollect(node: Node, forwardNodes: mutable.ArrayBuffer[Node],
      weightNodes: mutable.ArrayBuffer[Node], backwardNodes: mutable.ArrayBuffer[()=>Unit], liftNodes: mutable.Set[Backend.Sym],
      gradMap: GradMapWrapper,
      momentumMap: mutable.HashMap[Backend.Sym, TENSOR],
      transform: Backend.Exp => Backend.Exp) = node match {

    case Node(s, "tensor_add", tt::Backend.Const(anno:Anno)::(a:Backend.Sym)::(b:Backend.Sym)::_, _) =>
      implicit val pos = Adapter.oldSourceMap(s)
      // save forward op in forwardNodes
      forwardNodes += node
      // save backward op in backwardNodes
      (() => {
        Accumulate(gradMap(a), gradMap(s), anno); ()
      }) +=: backwardNodes
      (() => {
        Accumulate(gradMap(b), gradMap(s), anno); ()
      }) +=: backwardNodes

    // case Node(s, "tensor_sub", tt::Backend.Const(anno:Anno)::(a:Backend.Sym)::(b:Backend.Sym)::_, _) =>
    //   implicit val pos = Adapter.oldSourceMap(s)
    //   // save forward op in forwardNodes
    //   forwardNodes += node
    //   // save backward op in backwardNodes
    //   (() => {
    //     Accumulate(gradMap(a), gradMap(s), anno); ()
    //   }) +=: backwardNodes
    //   (() => {
    //     val b_grad = Neg(gradMap(s), anno)
    //     Accumulate(gradMap(b), b_grad, anno); ()
    //   }) +=: backwardNodes

    case Node(s, "tensor_mul", tt::Backend.Const(anno:Anno)::(a:Backend.Sym)::(b:Backend.Sym)::_, _) =>
      implicit val pos = Adapter.oldSourceMap(s)
      // save forward op in forwardNodes
      forwardNodes += node
      liftNodes += a
      liftNodes += b
      // save backward op in backwardNodes
      (() => {
        val a_tensor = new TENSOR(transform(a))
        val b_grad = Mul(a_tensor, gradMap(s), anno)
        Accumulate(gradMap(b), b_grad, anno); ()
      }) +=: backwardNodes
      (() => {
        val b_tensor = new TENSOR(transform(b))
        val a_grad = Mul(b_tensor, gradMap(s), anno)
        Accumulate(gradMap(a), a_grad, anno); ()
      }) +=: backwardNodes

    case Node(s, "tensor_div", tt::Backend.Const(anno:Anno)::(a:Backend.Sym)::(b:Backend.Sym)::_, _) =>
      implicit val pos = Adapter.oldSourceMap(s)
      // save forward op in forwardNodes
      forwardNodes += node
      liftNodes += ???
      // save backward op in backwardNodes
      (() => {
        ???
      }) +=: backwardNodes
      (() => {
        ???
      }) +=: backwardNodes

    case _ => super.aircopCollect(node, forwardNodes, weightNodes, backwardNodes, liftNodes, gradMap, momentumMap, transform)
  }

  override def printTensor(node: Node, graph: Graph): String = node match {
    case Node(s, op, Backend.Const(tt:TensorType)::Backend.Const(anno:Anno)::(a:Backend.Sym)::(b:Backend.Sym)::_, _) if binaryOps.contains(op) =>
      s"$s = $op($a, $b) (${symTensorShape(a, graph)}, ${symTensorShape(b, graph)})->${tt.toString}${if (anno != NAnno) s"  Anno: $anno" else ""}"
    case n => super.printTensor(n, graph)
  }
}


trait FixedSizeDistributedTensorOpsBinary extends FixedSizeDistributedTensorOpsBase {
  import FixedSizeDistributedTensorTypeLess._

  implicit class TensorOpsBinary[T:Numeric:Manifest](x: Rep[Tensor[T]]) {
    val self = tensor(x)

    def + (y: Rep[Tensor[T]])(implicit __pos: SourceContext, anno: Anno): Rep[Tensor[T]] = {
      val t = Add(self, tensor(y), anno)
      Wrap[Tensor[T]](t.x)
    }
    def + (y: Rep[Tensor[T]], anno: Anno)(implicit __pos: SourceContext): Rep[Tensor[T]] = {
      val t = Add(self, tensor(y), anno)
      Wrap[Tensor[T]](t.x)
    }

    def - (y: Rep[Tensor[T]])(implicit __pos: SourceContext, anno: Anno): Rep[Tensor[T]] = {
      val t = Sub(self, tensor(y), anno)
      Wrap[Tensor[T]](t.x)
    }
    def - (y: Rep[Tensor[T]], anno: Anno)(implicit __pos: SourceContext): Rep[Tensor[T]] = {
      val t = Sub(self, tensor(y), anno)
      Wrap[Tensor[T]](t.x)
    }

    def * (y: Rep[Tensor[T]])(implicit __pos: SourceContext, anno: Anno): Rep[Tensor[T]] = {
      val t = Mul(self, tensor(y), anno)
      Wrap[Tensor[T]](t.x)
    }
    def * (y: Rep[Tensor[T]], anno: Anno)(implicit __pos: SourceContext): Rep[Tensor[T]] = {
      val t = Mul(self, tensor(y), anno)
      Wrap[Tensor[T]](t.x)
    }

    def / (y: Rep[Tensor[T]])(implicit __pos: SourceContext, anno: Anno): Rep[Tensor[T]] = {
      val t = Div(self, tensor(y), anno)
      Wrap[Tensor[T]](t.x)
    }
    def / (y: Rep[Tensor[T]], anno: Anno)(implicit __pos: SourceContext): Rep[Tensor[T]] = {
      val t = Div(self, tensor(y), anno)
      Wrap[Tensor[T]](t.x)
    }
  }
}
