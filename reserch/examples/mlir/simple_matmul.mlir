// 简单matmul到BatchComp IR的转换示例
//
// 运行方式:
// mlir-opt simple_matmul.mlir \
//   --batch-comp-tile-scheduler="hardware=npu tile-size=16,16,16 algorithm=cpsat"
//
// 预期输出: BatchComp高层IR

module {
  // 单个matmul操作
  func.func @simple_matmul(%arg0: tensor<100x200xf32>, %arg1: tensor<200x300xf32>) -> tensor<100x300xf32> {
    // 输入: 标准TOSA matmul
    %0 = tosa.matmul %arg0, %arg1 : (tensor<100x200xf32>, tensor<200x300xf32>) -> tensor<100x300xf32>
    return %0 : tensor<100x300xf32>
  }

  // 预期转换结果（注释）:
  // func.func @simple_matmul(%arg0: tensor<100x200xf32>, %arg1: tensor<200x300xf32>) -> tensor<100x300xf32> {
  //   // Step 1: 生成tiles
  //   %tiles = batch_comp.generate_tiles %arg0, %arg1 {
  //     tile_size = [16, 16, 16],
  //     matrix_id = 0 : i32
  //   } : tensor<100x200xf32>, tensor<200x300xf32> -> !batch_comp.tileset
  //
  //   // Step 2: 调度batches (调用Python优化器)
  //   %schedule = batch_comp.schedule_batches %tiles {
  //     hardware = "npu",
  //     batch_size = 16 : i32,
  //     algorithm = "cpsat"
  //   } : !batch_comp.tileset -> !batch_comp.schedule
  //
  //   // Step 3: 执行调度
  //   %result = batch_comp.execute_schedule %schedule {
  //     hardware = "npu"
  //   } : !batch_comp.schedule -> tensor<100x300xf32>
  //
  //   return %result : tensor<100x300xf32>
  // }
}
