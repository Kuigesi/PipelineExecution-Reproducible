// Initial code:
def tensors_06(x0: Int): Int /* x1 */ = {
  val x2 = Tensor1(List(100), ((x4: Int) /* x3 */ => 1 /* x3 */))
  val x5 = Tensor1(List(100), ((x6: Int) /* x7 */ => 2 * seq_apply(x6, 0) /* x7 */))
  val x8 = Tensor1(List(100), ((x9: Int) /* x10 */ => tensor_apply(x2, x9) + tensor_apply(x5, x9) /* x10 */))
  val x11 = Sum(List(100), ((x12: Int) /* x13 */ => tensor_apply(x8, x12) /* x13 */)) / 100
  println(Sum(List(100), ((x14: Int) /* x15 */ => tensor_apply(x8, x14) /* x15 */)) / 100)/* val x16 = "CTRL":x1 */
  println(Sum(List(100), ({ x18: Int /* x19 */ =>
    val x17 = tensor_apply(x8, x18)
    x17 * x17 /* x19 */
  })) / 100 - x11 * x11)/* val x20 = "CTRL":x16 */
  0 /* x20 */
}
// After Tensor lowering:
def tensors_06(x0: Int): Int /* x1 */ = {
  val x2 = Tensor1(List(100), ((x4: Int) /* x3 */ => 1 /* x3 */))
  val x5 = Tensor1(List(100), ((x6: Int) /* x7 */ => 2 * seq_apply(x6, 0) /* x7 */))
  val x8 = Tensor1(List(100), ((x9: Int) /* x10 */ => tensor_apply(x2, x9) + tensor_apply(x5, x9) /* x10 */))
  val x11 = Sum(List(100), ((x12: Int) /* x13 */ => tensor_apply(x8, x12) /* x13 */)) / 100
  println(Sum(List(100), ((x14: Int) /* x15 */ => tensor_apply(x8, x14) /* x15 */)) / 100)/* val x16 = "CTRL":x1 */
  println(Sum(List(100), ({ x18: Int /* x19 */ =>
    val x17 = tensor_apply(x8, x18)
    x17 * x17 /* x19 */
  })) / 100 - x11 * x11)/* val x20 = "CTRL":x16 */
  0 /* x20 */
}
// After Tensor fusion V:
def tensors_06(x0: Int): Int /* x1 */ = {
  val x2 = Sum(List(100), ((x3: Int) /* x4 */ => 1 + 2 * seq_apply(x3, 0) /* x4 */)) / 100
  println(Sum(List(100), ((x5: Int) /* x6 */ => 1 + 2 * seq_apply(x5, 0) /* x6 */)) / 100)/* val x7 = "CTRL":x1 */
  println(Sum(List(100), ({ x9: Int /* x10 */ =>
    val x8 = 1 + 2 * seq_apply(x9, 0)
    x8 * x8 /* x10 */
  })) / 100 - x2 * x2)/* val x11 = "CTRL":x7 */
  0 /* x11 */
}
// After Tensor fusion H:
def tensors_06(x0: Int): Int /* x1 */ = {
  val x2 = Sum(List(100), ((x3: Int) /* x4 */ => 1 + 2 * seq_apply(x3, 0) /* x4 */)) / 100
  println(Sum(List(100), ((x5: Int) /* x6 */ => 1 + 2 * seq_apply(x5, 0) /* x6 */)) / 100)/* val x7 = "CTRL":x1 */
  println(Sum(List(100), ({ x9: Int /* x10 */ =>
    val x8 = 1 + 2 * seq_apply(x9, 0)
    x8 * x8 /* x10 */
  })) / 100 - x2 * x2)/* val x11 = "CTRL":x7 */
  0 /* x11 */
}
// After Multiloop/Builder lowering:
def tensors_06(x0: Int): Int /* x1 */ = {
  val x2 = SumBuilder1(0)/* val x2 = Const(STORE):x1 */
  val x3 = forloops(List(100), ({ x5: Int /* x6 */ =>
    val x4 = sum_builder_add(x2, x5, 1 + 2 * seq_apply(x5, 0))/* val x4 = x2:x6 */
    /* x4 */
  })/* val x7 = x2:x2 */)/* val x3 = x2:x7 */
  val x8 = SumBuilder1(0)/* val x8 = Const(STORE):x2 */
  val x9 = forloops(List(100), ({ x11: Int /* x13 */ =>
    val x10 = 1 + 2 * seq_apply(x11, 0)
    val x12 = sum_builder_add(x8, x11, x10 * x10)/* val x12 = x8:x13 */
    /* x12 */
  })/* val x14 = x8:x8x1 */)/* val x9 = x8:x14x1 */
  val x15 = SumBuilder1(0)/* val x15 = Const(STORE):x8x1x2 */
  val x16 = forloops(List(100), ({ x18: Int /* x19 */ =>
    val x17 = sum_builder_add(x15, x18, 1 + 2 * seq_apply(x18, 0))/* val x17 = x15:x19 */
    /* x17 */
  })/* val x20 = x15:x15x1 */)/* val x16 = x15:x20x1 */
  val x21 = sum_builder_res(x15)/* val x22 = x15:x16x1 */ / 100
  println(sum_builder_res(x2)/* val x23 = x2:x3 */ / 100)/* val x24 = "CTRL":x1 */
  println(sum_builder_res(x8)/* val x25 = x8:x9x1 */ / 100 - x21 * x21)/* val x26 = "CTRL":x24x1 */
  0 /* x26x22x25x23 */
}
// After Tensor fusion H2:
def tensors_06(x0: Int): Int /* x1 */ = {
  val x2 = SumBuilder1(0)/* val x2 = Const(STORE):x1 */
  val x3 = SumBuilder1(0)/* val x3 = Const(STORE):x2 */
  val x4 = SumBuilder1(0)/* val x4 = Const(STORE):x3 */
  val x5 = forloops(List(100), ({ x7: Int /* x9 */ =>
    val x6 = 1 + 2 * seq_apply(x7, 0)
    val x8 = sum_builder_add(x2, x7, x6)/* val x8 = x2:x9 */
    val x10 = sum_builder_add(x3, x7, x6 * x6)/* val x10 = x3:x9 */
    val x11 = sum_builder_add(x4, x7, x6)/* val x11 = x4:x9 */
    /* x11x10x8 */
  })/* val x12 = x2,x3,x4:x4 */)/* val x5 = x2,x3,x4:x12 */
  val x13 = sum_builder_res(x4)/* val x14 = x4:x5 */ / 100
  println(sum_builder_res(x2)/* val x15 = x2:x5 */ / 100)/* val x16 = "CTRL":x1 */
  println(sum_builder_res(x3)/* val x17 = x3:x5 */ / 100 - x13 * x13)/* val x18 = "CTRL":x16x1 */
  0 /* x18x15x14x17 */
}
// After MultiDim foreach lowering:
def tensors_06(x0: Int): Int /* x1 */ = {
  var x2 = 0
  var x3 = 0
  var x4 = 0
  val x5 = forloop(100, ({ x7: Int /* x9 */ =>
    val x6 = 1 + 2 * x7
    x2 = x2/* x2:x9 */ + x6
    x3 = x3/* x3:x9 */ + x6 * x6
    x4 = x4/* x4:x9 */ + x6
    /* x12x13x14 */
  })/* val x15 = x2,x3,x4:x4 */)/* val x5 = x2,x3,x4:x15 */
  val x16 = x4/* val x17 = x4:x5 */ / 100
  println(x2/* val x18 = x2:x5 */ / 100)/* val x19 = "CTRL":x1 */
  println(x3/* val x20 = x3:x5 */ / 100 - x16 * x16)/* val x21 = "CTRL":x19x1 */
  0 /* x21x18x17x20 */
}
