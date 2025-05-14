// =============================================================================
//
// This file lowers the ONNX HardSwish Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//


#include "mlir/IR/BuiltinTypes.h"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// ONNXHardSwishOpLowering
//===----------------------------------------------------------------------===//

struct ONNXHardSwishOpLowering : public OpConversionPattern<ONNXHardSwishOp> {
  ONNXHardSwishOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXHardSwishOp hardSwishOp,
      ONNXHardSwishOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = hardSwishOp.getOperation();
    Location loc = ONNXLoc<ONNXHardSwishOp>(op);
    ValueRange operands = adaptor.getOperands();
    Value inputOperand = adaptor.getX();

    // Builder for Krnl, Math, and MemRef operations.
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);

    // --- Shape Inference ---
    // For element-wise operations like HardSwish, the output shape is the same
    // as the input shape.
    ONNXUnaryOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // --- Memory Allocation ---
    // Allocate memory for the output tensor.
    // The type of the output tensor will be the same as the input tensor.
    Value alloc = allocForONNXOp<ONNXHardSwishOp>(
        hardSwishOp, rewriter, *getTypeConverter(), shapeHelper)[0];
    MemRefType outputMemRefType = mlir::cast<MemRefType>(alloc.getType());
    Type elementType = outputMemRefType.getElementType();

    // --- Constants for HardSwish formula ---
    // HardSwish formula: y = x * max(0, min(6, x + 3)) / 6
    // We need constants: 0.0, 3.0, 6.0.
    // The division by 6 can be a multiplication by 1.0/6.0.
    Value constZero = create.math.constant(elementType, 0.0);
    Value constThree = create.math.constant(elementType, 3.0);
    Value constSix = create.math.constant(elementType, 6.0);
    // Using multiplication by 1.0/6.0 can sometimes be more stable or allow
    // for fma instructions, though div is also fine.
    Value constOneSixth = create.math.constant(elementType, 1.0 / 6.0);

    // --- Krnl Loops for Element-wise Computation ---
    int inputRank = outputMemRefType.getRank();
    ValueRange loopDefine = create.krnl.defineLoops(inputRank);

    SmallVector<IndexExpr, 4> lbs(inputRank, LiteralIndexExpr(0));
    SmallVector<IndexExpr, 4> ubs;
    for (int i = 0; i < inputRank; ++i) {
      ubs.emplace_back(create.krnlIE.getShapeAsSymbol(inputOperand, i));
    }

    // Create the nested loops that iterate over each element of the tensor.
    create.krnl.iterateIE(loopDefine, loopDefine, lbs, ubs,
        [&](KrnlBuilder &krnlBuilder, ValueRange loopIndices) {
          // `loopIndices` contains the current multi-dimensional index.
          // Create a new MathBuilder instance for this specific scope if needed,
          // or reuse the outer `create.math`.
          MathBuilder math(krnlBuilder.getBuilder(), krnlBuilder.getLoc());

          // Load the current input element x.
          Value x = krnlBuilder.load(inputOperand, loopIndices);

          // Apply HardSwish formula: y = x * max(0, min(6, x + 3)) / 6
          // 1. temp1 = x + 3
          Value temp1 = math.add(x, constThree);
          // 2. temp2 = min(6, temp1)
          //    Note: math.min/max typically map to arith.minf/maxf for floats.
          Value temp2 = math.min(constSix, temp1);
          // 3. temp3 = max(0, temp2)
          Value temp3 = math.max(constZero, temp2);
          // 4. temp4 = x * temp3
          Value temp4 = math.mul(x, temp3);
          // 5. y = temp4 / 6  (or temp4 * (1/6))
          Value y = math.mul(temp4, constOneSixth); // Using multiply by 1.0/6.0

          // Store the computed result y into the output tensor.
          krnlBuilder.store(y, alloc, loopIndices);
        });

    // Replace the original ONNXHardSwishOp with the KRNL operations
    // that produce the result in `alloc`.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helper function to populate the lowering pattern.
//===----------------------------------------------------------------------===//

void populateLoweringONNXHardSwishOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXHardSwishOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir