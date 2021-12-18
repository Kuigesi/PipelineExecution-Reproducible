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


class PipelineConvolutionTest extends TutorialFunSuite {
  val under = "transformer/pipeline_conv/"

  abstract class CompilerCPipelineConvolution[A: Manifest, B: Manifest] extends CompilerC[A,B] with FixedSizeDistributedTensorOps { q =>

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
      new DistributeTensor2MPI_NCCL {},
      new DistributeTensorMemoryOpt {}
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
  val matrixdim = 128
  val outchannel = 256
  val kernelsize = 17
  val padding = (kernelsize -1)/2
  val pipeline = 4
  val iternum = 40
  val ndim = 4

  test("conv-data2") {
    val driver = new CompilerCPipelineConvolution[Int, Unit] {
      import FixedSizeDistributedTensorTypeLess._

      @virtualize
      def snippet(arg: Rep[Int]): Rep[Unit] = {

        val model = module {
          val input = Tensor.input[Float](shape=Seq(ndim,1,matrixdim,matrixdim), index=0, devices=List(GPU(0), GPU(1)))
          implicit val anno = input.anno
          val weight = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight2 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight3 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight4 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight5 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight6 = Tensor.weight[Float](Seq(1,outchannel,kernelsize,kernelsize))
                                            // padding   stride    dilation

          val weight7 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight8 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight9 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight10 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight11 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight12 = Tensor.weight[Float](Seq(1,outchannel,kernelsize,kernelsize))
          val params = ConvParam(1.0f, 0.0f, Seq(padding, padding), Seq(1, 1), Seq(1, 1))
          val recv = input.conv(weight, params).relu.conv(weight2, params).relu.conv(weight3, params).relu.conv(weight4, params).relu.conv(weight5, params).relu.conv(weight6, params).relu
          recv.conv(weight7, params).relu.conv(weight8, params).relu.conv(weight9, params).relu.conv(weight10, params).relu.conv(weight11, params).relu.conv(weight12, params).relu
        }
        model.train(iternum);
        ()
      }
    }
    driver.setLogPath(log_path("conv-data2", ".cu"))
    writeFile(prefix+under+"conv-data2/" +"conv-data2.test.cu", indent(driver.code))
  }

  test("conv-data4") {
    val driver = new CompilerCPipelineConvolution[Int, Unit] {
      import FixedSizeDistributedTensorTypeLess._

      @virtualize
      def snippet(arg: Rep[Int]): Rep[Unit] = {

        val model = module {
          val input = Tensor.input[Float](shape=Seq(ndim,1,matrixdim,matrixdim), index=0, devices=List(GPU(0), GPU(1), GPU(2), GPU(3)))
          implicit val anno = input.anno
          val weight = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight2 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight3 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight4 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight5 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight6 = Tensor.weight[Float](Seq(1,outchannel,kernelsize,kernelsize))
                                            // padding   stride    dilation

          val weight7 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight8 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight9 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight10 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight11 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight12 = Tensor.weight[Float](Seq(1,outchannel,kernelsize,kernelsize))
          val params = ConvParam(1.0f, 0.0f, Seq(padding, padding), Seq(1, 1), Seq(1, 1))
          val recv = input.conv(weight, params).relu.conv(weight2, params).relu.conv(weight3, params).relu.conv(weight4, params).relu.conv(weight5, params).relu.conv(weight6, params).relu
          recv.conv(weight7, params).relu.conv(weight8, params).relu.conv(weight9, params).relu.conv(weight10, params).relu.conv(weight11, params).relu.conv(weight12, params).relu
        }
        model.train(iternum);
        ()
      }
    }

    driver.setLogPath(log_path("conv-data4", ".cu"))
    writeFile(prefix+under+"conv-data4/" +"conv-data4.test.cu", indent(driver.code))
  }

  test("conv-nopipeline") {
    val driver = new CompilerCPipelineConvolution[Int, Unit] {
      import FixedSizeDistributedTensorTypeLess._

      @virtualize
      def snippet(arg: Rep[Int]): Rep[Unit] = {

        val model = module (MAnno(List(GPU(0), GPU(1)))) {
          val input = Tensor.input[Float](shape=Seq(ndim,1,matrixdim,matrixdim), index=0, devices=List(GPU(0), GPU(1)))
          implicit val anno = input.anno
          val weight = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight2 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight3 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight4 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight5 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight6 = Tensor.weight[Float](Seq(1,outchannel,kernelsize,kernelsize))
                                            // padding   stride    dilation
          val params = ConvParam(1.0f, 0.0f, Seq(padding, padding), Seq(1, 1), Seq(1, 1))
          input.conv(weight, params).relu.conv(weight2, params).relu.conv(weight3, params).relu.conv(weight4, params).relu.conv(weight5, params).relu.conv(weight6, params).relu
        }
        model.train(iternum);
        val model2 = module (MAnno(List(GPU(2), GPU(3)), islastmodule = true)) {
          val recv = model.recv(NAnno)
          implicit val anno = recv.anno
          val weight = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight2 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight3 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight4 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight5 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight6 = Tensor.weight[Float](Seq(1,outchannel,kernelsize,kernelsize))
          val params = ConvParam(1.0f, 0.0f, Seq(padding, padding), Seq(1, 1), Seq(1, 1))
          recv.conv(weight, params).relu.conv(weight2, params).relu.conv(weight3, params).relu.conv(weight4, params).relu.conv(weight5, params).relu.conv(weight6, params).relu
        }
        model2.train(iternum);
        ()
      }
    }

    driver.setLogPath(log_path("conv-nopipeline", ".cu"))
    writeFile(prefix+under+"conv-nopipeline/" +"conv-nopipeline.test.cu", indent(driver.code))
  }

  test("conv-4pipeline") {
    val driver = new CompilerCPipelineConvolution[Int, Unit] {
      import FixedSizeDistributedTensorTypeLess._

      @virtualize
      def snippet(arg: Rep[Int]): Rep[Unit] = {
        val model = module (KAnno(pipeline, List(GPU(0), GPU(1)))) {
          val input = Tensor.input[Float](shape=Seq(ndim,1,matrixdim,matrixdim), index=0, devices=List(GPU(0), GPU(1)))
          implicit val anno = input.anno
          val weight = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight2 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight3 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight4 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight5 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight6 = Tensor.weight[Float](Seq(1,outchannel,kernelsize,kernelsize))
                                            // padding   stride    dilation
          val params = ConvParam(1.0f, 0.0f, Seq(padding, padding), Seq(1, 1), Seq(1, 1))
          input.conv(weight, params).relu.conv(weight2, params).relu.conv(weight3, params).relu.conv(weight4, params).relu.conv(weight5, params).relu.conv(weight6, params).relu
        }
        model.train(iternum);
        val model2 = module (KAnno(pipeline, List(GPU(2), GPU(3)), islastmodule = true)) {
          val recv = model.recv(NAnno)
          implicit val anno = recv.anno
          val weight = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight2 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight3 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight4 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight5 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight6 = Tensor.weight[Float](Seq(1,outchannel,kernelsize,kernelsize))
          val params = ConvParam(1.0f, 0.0f, Seq(padding, padding), Seq(1, 1), Seq(1, 1))
          recv.conv(weight, params).relu.conv(weight2, params).relu.conv(weight3, params).relu.conv(weight4, params).relu.conv(weight5, params).relu.conv(weight6, params).relu
        }
        model2.train(iternum);
        ()
      }
    }
    driver.setLogPath(log_path("conv-4pipeline", ".cu"))
    writeFile(prefix+under+"conv-4pipeline/" +"conv-4pipeline.test.cu", indent(driver.code))
  }

  test("conv-8pipeline") {
    val driver = new CompilerCPipelineConvolution[Int, Unit] {
      import FixedSizeDistributedTensorTypeLess._

      @virtualize
      def snippet(arg: Rep[Int]): Rep[Unit] = {
        val model = module (KAnno(8, List(GPU(0), GPU(1)))) {
          val input = Tensor.input[Float](shape=Seq(ndim,1,matrixdim,matrixdim), index=0, devices=List(GPU(0), GPU(1)))
          implicit val anno = input.anno
          val weight = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight2 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight3 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight4 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight5 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight6 = Tensor.weight[Float](Seq(1,outchannel,kernelsize,kernelsize))
                                            // padding   stride    dilation
          val params = ConvParam(1.0f, 0.0f, Seq(padding, padding), Seq(1, 1), Seq(1, 1))
          input.conv(weight, params).relu.conv(weight2, params).relu.conv(weight3, params).relu.conv(weight4, params).relu.conv(weight5, params).relu.conv(weight6, params).relu
        }
        model.train(iternum);
        val model2 = module (KAnno(8, List(GPU(2), GPU(3)), islastmodule = true)) {
          val recv = model.recv(NAnno)
          implicit val anno = recv.anno
          val weight = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight2 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight3 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight4 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight5 = Tensor.weight[Float](Seq(outchannel,1,kernelsize,kernelsize))
          val weight6 = Tensor.weight[Float](Seq(1,outchannel,kernelsize,kernelsize))
          val params = ConvParam(1.0f, 0.0f, Seq(padding, padding), Seq(1, 1), Seq(1, 1))
          recv.conv(weight, params).relu.conv(weight2, params).relu.conv(weight3, params).relu.conv(weight4, params).relu.conv(weight5, params).relu.conv(weight6, params).relu
        }
        model2.train(iternum);
        ()
      }
    }
    driver.setLogPath(log_path("conv-8pipeline", ".cu"))
    writeFile(prefix+under+"conv-8pipeline/" +"conv-8pipeline.test.cu", indent(driver.code))
  }
}
