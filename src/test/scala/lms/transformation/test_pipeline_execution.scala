package lms
package transformation.tensor

import scala.annotation.implicitNotFound
import scala.collection.immutable.Seq
import lms.core.virtualize
import macros.SourceContext

import lms.core._
import lms.core.stub._
import lms.thirdparty.{CCodeGenLibs, CCodeGenMPI, CCodeGenNCCLOps, CCodeGenCUDNN, CCodeGenScannerOps}
import lms.thirdparty.array_computation.{CCodeGenCBLASOps, CCodeGenCudaOps, CCodeGenCuBLAS}

import Backend._


class PipelineExecutionTensorTest extends TutorialFunSuite {
  val under = "transformer/pipeline_execution/"

  abstract class CompilerCPipelineExecution[A: Manifest, B: Manifest] extends CompilerC[A,B] with FixedSizeDistributedTensorOps { q =>

    override val codegen = new DslGenCPP with CCodeGenLibs with CCodeGenCBLASOps with
        CCodeGenCudaOps with CCodeGenNCCLOps with CCodeGenMPI with CCodeGenCuBLAS with CCodeGenCUDNN with CCodeGenScannerOps {
      val IR: q.type = q

      override def mayInline(n: Node): Boolean = n match {
        case Node(_, s, _, _) if s.startsWith("tensor_") || s.startsWith("tensors_") => false
        case _ => super.mayInline(n)
      }
    }

    override val passes = List(
      new DistributeTensorDimName {},
      new DistributeTensorAIRCoP {},
      new Canonicalize {},
      new DistributeTensorAIRCoPSpatial {},
      new DistributeTensor2MPI_NCCL {}
      )

    var log_path: String = ""
    def setLogPath(path: String) { log_path = path }

    override def transform(graph: Graph): Graph = {
      logGraph(show_graph(graph), log_path)
      super.transform(graph)
    }

    override def transformOnePass(pass: Transformer, index: Int, graph: Graph) = {
      val new_graph = pass.transform(graph)
      if (log_path == "") throw new Exception("should set log_path first")
      logGraph(show_graph(new_graph), log_path, index, pass.name)
      new_graph
    }

    def show_graph(graph: Graph): String = {
      // return a string representation of the graph
      val source = new java.io.ByteArrayOutputStream()
      val stream = new java.io.PrintStream(source)
      stream.println("==================")
      for (node <- graph.nodes)
        stream.println(showTensor(node, graph))
      stream.println(graph.block)
      stream.println("==================")
      source.toString
    }
  }

  test("mult") {
    val driver = new CompilerCPipelineExecution[Int, Unit] {
      import FixedSizeDistributedTensorTypeLess._

      @virtualize
      def snippet(arg: Rep[Int]): Rep[Unit] = {

        val model = module (KAnno(3, List(GPU(0), GPU(1)))) {
          val input = Tensor.input[Float](shape=Seq(32,32), name="input1", splitDim=0, splitTo=List(GPU(0), GPU(1)))
          //print("\n <<<<<< input generate >>>>\n")
          // this is still hacky :(
          implicit val anno = input.anno
          val weight = Tensor.weight[Float](Seq(32, 32), tensorName=Some("weight"))
          input * weight
        }
        model.train(10);
        val model2 = module (KAnno(3, List(GPU(2), GPU(3)), islastmodule = true)) {
          val input2 = Tensor.input[Float](shape=Seq(32,32), name="input2", splitDim=0, splitTo=List(GPU(2), GPU(3)))
          // this is still hacky :(
          implicit val anno2 = input2.anno
          //val weight2 = Tensor.weight[Float](Seq(32, 32), tensorName=Some("weight"))
          // input2 * weight2
         input2  * model
          //input2 * model
          //input2
        }
        model2.train(10);
        //model2.test("loss");*/
        /*
        val model = module {
          val input = Tensor.input[Float](shape=Seq(32,32), name="input", splitDim=0, splitTo=List(GPU(0), GPU(1)))
          // this is still hacky :(
          implicit val anno = input.anno
          val weight = Tensor.weight[Float](Seq(32, 32), tensorName=Some("weight"))
          input * weight
        }
        model.train(10);*/
        ()
      }
    }
    // // setPath(log_path(mult, suffix))
   // // driver.set
    //driver.setLogPath(log_path("mult", ".cu"))
    //writeFile(prefix+under+"mult/" +"mult.output.cu", indent(driver.code))
    checkWithLogPath("mult", driver.code, "cu", driver.setLogPath)
  }
}
