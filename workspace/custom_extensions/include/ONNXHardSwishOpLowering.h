// filepath: /path/to/your/onnx-mlir/src/Conversion/ONNXToKrnl/ONNXHardSwishOpLowering.hpp
#pragma once
#include "mlir/Pass/Pass.h" // For RewritePatternSet
#include "mlir/Transforms/DialectConversion.h" // For TypeConverter

namespace mlir {
class MLIRContext;
}

namespace onnx_mlir {
void populateLoweringONNXHardSwishOpPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *ctx);
} // namespace onnx_mlir