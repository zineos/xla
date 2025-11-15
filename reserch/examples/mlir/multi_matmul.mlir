// 多matmul跨矩阵批处理优化示例
//
// 运行方式:
// mlir-opt multi_matmul.mlir \
//   --batch-comp-tile-scheduler="hardware=npu tile-size=16,16,16 algorithm=cpsat enable-multi-matmul=true"
//
// 预期输出: 跨多个matmul的联合优化BatchComp IR

module {
  // 多个matmul操作在同一个函数中
  func.func @multi_matmul(
    %A0: tensor<100x200xf32>, %B0: tensor<200x300xf32>,
    %A1: tensor<48x64xf32>, %B1: tensor<64x48xf32>,
    %A2: tensor<35x50xf32>, %B2: tensor<50x40xf32>
  ) -> (tensor<100x300xf32>, tensor<48x48xf32>, tensor<35x40xf32>) {

    // Matmul 0: 中等大小
    %C0 = tosa.matmul %A0, %B0 : (tensor<100x200xf32>, tensor<200x300xf32>) -> tensor<100x300xf32>

    // Matmul 1: 小矩阵
    %C1 = tosa.matmul %A1, %B1 : (tensor<48x64xf32>, tensor<64x48xf32>) -> tensor<48x48xf32>

    // Matmul 2: 不规则矩阵
    %C2 = tosa.matmul %A2, %B2 : (tensor<35x50xf32>, tensor<50x40xf32>) -> tensor<35x40xf32>

    return %C0, %C1, %C2 : tensor<100x300xf32>, tensor<48x48xf32>, tensor<35x40xf32>
  }

  // 预期转换结果（注释）:
  //
  // func.func @multi_matmul(...) -> (...) {
  //   // 为每个matmul生成tiles
  //   %tiles0 = batch_comp.generate_tiles %A0, %B0 {
  //     tile_size = [16, 16, 16], matrix_id = 0 : i32
  //   } : ... -> !batch_comp.tileset
  //
  //   %tiles1 = batch_comp.generate_tiles %A1, %B1 {
  //     tile_size = [16, 16, 16], matrix_id = 1 : i32
  //   } : ... -> !batch_comp.tileset
  //
  //   %tiles2 = batch_comp.generate_tiles %A2, %B2 {
  //     tile_size = [16, 16, 16], matrix_id = 2 : i32
  //   } : ... -> !batch_comp.tileset
  //
  //   // 合并所有tiles进行联合调度
  //   // 注意: 实际实现中会在Pass内部合并，这里展示的是概念
  //
  //   // 跨矩阵批处理调度
  //   %schedule0 = batch_comp.schedule_batches %tiles0 {
  //     hardware = "npu", batch_size = 16 : i32, algorithm = "cpsat"
  //   } : !batch_comp.tileset -> !batch_comp.schedule
  //
  //   %schedule1 = batch_comp.schedule_batches %tiles1 {
  //     hardware = "npu", batch_size = 16 : i32, algorithm = "cpsat"
  //   } : !batch_comp.tileset -> !batch_comp.schedule
  //
  //   %schedule2 = batch_comp.schedule_batches %tiles2 {
  //     hardware = "npu", batch_size = 16 : i32, algorithm = "cpsat"
  //   } : !batch_comp.tileset -> !batch_comp.schedule
  //
  //   // 执行
  //   %C0 = batch_comp.execute_schedule %schedule0 {hardware = "npu"} : ...
  //   %C1 = batch_comp.execute_schedule %schedule1 {hardware = "npu"} : ...
  //   %C2 = batch_comp.execute_schedule %schedule2 {hardware = "npu"} : ...
  //
  //   return %C0, %C1, %C2 : ...
  // }
  //
  // 关键优势:
  // - 来自不同matmul的tiles可以组合到同一个batch中
  // - 提高硬件利用率（减少NOPs）
  // - 保持K-dimension累加正确性
}
