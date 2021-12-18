package lms.transformation.tensor

import scala.annotation.implicitNotFound
import scala.collection._

import lms.core._
import lms.core.stub._
import lms.collection.mutable._
import lms.macros.SourceContext
import lms.thirdparty.array_computation.{ArrayCPUTypeLess, CUDATypeLess, CUBLASTypeLess}
import lms.transformation.util.DataStructure

import Backend._

abstract class DistributeTensorMemoryOpt extends Transformer {
  import DataStructure._
  override val name = "DistributeTensorMemoryOpt"

  //import Backend._
  import PrimitiveTypeLess._
  import ArrayTypeLess._
  import ArrayCPUTypeLess._
  import FixedSizeDistributedTensorTypeLess._
  import CUDATypeLess._
  import CUBLASTypeLess._

  override def scheduleBlock_[T](y: Block, extra: Backend.Sym*)(f: (List[Backend.Sym], Seq[Node], Seq[Node], Block) => T): T = {

    // when entering a block, we get more bound variables (from the block and possibly supplimented
    // via `extra)
    val path1 = y.bound ++ extra.toList ++ path

    // a node is available if all bound vars
    // it depends on are in scope

    def available_2(d: Node) =
      bound.hm(d.n) -- path1 - d.n == Set()

    // find out which nodes are reachable on a
    // warm path (not only via if/else branches)
    val g = new Graph(inner, y, null)

    val malloc = new mutable.HashSet[Backend.Sym]()
    for (d <- g.nodes.reverseIterator) {

      if (malloc contains d.n) {
        malloc ++= syms(d)
      }
      else {
      d match {
        case Node(s, "lib-function", Backend.Const("CUDA_CALL")::Backend.Const(pkeys)::(m:Backend.Sym)::_,_) => {
          Adapter.oldDefsCache.get(m) match {
            case Some(n:Node) => n match {
                                    case Node(s, "lib-function", Backend.Const("cudaMalloc")::_,_) => malloc += d.n
                                    case _ =>
                                  }
            case None =>
          }
          //malloc ++= syms(d)
        }
        case Node(s, "lib-function", Backend.Const("cudaMalloc")::_,_) => {
          malloc += d.n
          malloc ++= syms(d)
        }
        case _ =>
      }
      }
    }
    //if (!malloc.isEmpty) {
    //  val b = malloc.toList.sortWith(_.n < _.n)
    //  print(b+"\n")
    //}

    def available(d: Node) =
      (bound.hm(d.n) -- path1 - d.n == Set()) || {
        malloc contains d.n
      }

    val reach = new mutable.HashSet[Backend.Sym]
    val reachInner = new mutable.HashSet[Backend.Sym]
    reach ++= y.used

    for (d <- g.nodes.reverseIterator) {

      if (reach contains d.n) {
        if (available(d)) {
          // node will be sched here, don't follow if branches!
          // other statement will be scheduled in an inner block
          for ((e:Backend.Sym,f) <- symsFreq(d))
            if (f > 0.5) reach += e else reachInner += e
        } else {
          // QUESTION(feiw): why we don't split via frequency here?
          reach ++= hardSyms(d)
        }
      }
      if (reachInner.contains(d.n)) {
        reachInner ++= hardSyms(d)
      }
    }

    def scheduleHere(d: Node) =
      available(d) && reach(d.n)

    var outer1 = Seq[Node]()
    var inner1 = Seq[Node]()

    val extraThroughSoft = new mutable.HashSet[Backend.Sym]
    for (n <- inner.reverseIterator) {
      if (reach(n.n) || extraThroughSoft(n.n)) {
        if (available(n)) {
          outer1 = n +: outer1
          if (!reach(n.n)) // if added through soft deps, hards needs to be added as well
            extraThroughSoft ++= syms(n)
          else
            extraThroughSoft ++= n.eff.sdeps
        } else {
          inner1 = n +: inner1
        }
      } else if (reachInner(n.n)) {
        inner1 = n +: inner1
      }
    }

    f(path1, inner1, outer1, y)
  }

  override def transform(graph: Graph): Graph = {
    assert (g == null)
    g = new GraphBuilderOpt()
    Adapter.g = g

    try {
      super.transform(graph)
    } finally {
      g = null; Adapter.g = null
    }
  }
}
