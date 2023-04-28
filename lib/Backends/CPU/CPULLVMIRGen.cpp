/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "CPULLVMIRGen.h"

#include "glow/IR/Instrs.h"
#include "glow/LLVMIRCodeGen/LLVMBackend.h"
#include "glow/Quantization/Base/Base.h"

using namespace glow;
using llvm::cast;

CPULLVMIRGen::CPULLVMIRGen(const IRFunction *F,
                           AllocationsInfo &allocationsInfo,
                           std::string mainEntryName, llvm::StringRef libjitBC)
    : LLVMIRGen(F, allocationsInfo, mainEntryName, libjitBC) {}

CPULLVMIRGen::CPULLVMIRGen(const IRFunction *F,
                           AllocationsInfo &allocationsInfo,
                           std::string mainEntryName, llvm::StringRef libjitBC,
                           llvm::ArrayRef<llvm::MemoryBufferRef> objectRegistry)
    : LLVMIRGen(F, allocationsInfo, mainEntryName, libjitBC, objectRegistry) {}

void CPULLVMIRGen::generateLLVMIRForModule(llvm::IRBuilder<> &builder) {
  // TODO: Add here any backend specific logic.
  LLVMIRGen::generateLLVMIRForModule(builder);
}

void CPULLVMIRGen::generateLLVMIRForInstr(llvm::IRBuilder<> &builder,
                                          const glow::Instruction *I) {
  setCurrentDebugLocation(builder, I);
  assert(!canBePartOfDataParallelKernel(I) &&
         "data parallel instructions are not handled here");
  // Perform any backend-specific code generation here and delegate everything
  // else to LLVMIRGen.
  switch (I->getKind()) {
  case Kinded::Kind::CPUConvDKKC8InstKind: {
    auto *CI = cast<CPUConvDKKC8Inst>(I);
    auto *dest = CI->getDest();
    auto *src = CI->getSrc();
    auto *filter = CI->getFilter();
    auto *bias = CI->getBias();
    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);
    auto *filterPtr = emitValueAddress(builder, filter);
    auto *biasPtr = emitValueAddress(builder, bias);

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);
    auto *filterDims = emitValueDims(builder, filter);
    auto *biasDims = emitValueDims(builder, bias);

    auto *kernels = emitConstDimTArray(builder, CI->getKernels());
    auto *strides = emitConstDimTArray(builder, CI->getStrides());
    auto *pads = emitConstDimTArray(builder, CI->getPads());
    auto *group = emitConstDimT(builder, CI->getGroup());

    size_t inChannels = src->dims()[3];
    size_t outChannels = dest->dims()[3];

    // Select a method for iterating on the image in the pixel (filter-first, or
    // input-first). Perform convolutions with a high channel count by scanning
    // the input image multiple times, once for each filter entry. Scan images
    // with a low channel count by scanning the image once because the filter
    // scan will fall in the cache.
    bool pixelScanFirst = (inChannels < 16);

    // The number of float8 registers that we use to process the depth channel.
    unsigned numDepthRegs = (pixelScanFirst ? 8 : 2);
    // The number of y pixels to process at once.
    unsigned sizeGroupY = (pixelScanFirst ? 1 : 5);

    // When producing output pixels process this many times of depth-strips,
    // where each chunk is float8 * numDepthRegs. This is a form of tiling. It's
    // profitable to scan multiple depth-strips of the filter if the scanned
    // memory fits in the cahce and does not get evicted before the next
    // iteration. By increasing the number strips (and using more cache memory)
    // we reduce the number of times that we iterate over the input. However, we
    // also increase the pressure on the cache that has to store the filter so
    // we can't process too many strips at once.
    unsigned depthStrips = 1;
    unsigned stripSize = 8 * numDepthRegs * inChannels;
    unsigned tileSize = 16384;
    // Increase the number of strips until we reach the output-tensor depth size
    // or until we exceed some threashold.
    while (2 * depthStrips * stripSize <= tileSize &&
           2 * depthStrips * numDepthRegs * 8 <= outChannels / CI->getGroup() &&
           depthStrips < 8) {
      depthStrips *= 2;
    }

    auto *pixelScanFirstVal = emitConstI32(builder, pixelScanFirst);
    auto *numDepthRegsVal = emitConstI32(builder, numDepthRegs);
    auto *sizeGroupYVal = emitConstI32(builder, sizeGroupY);
    auto *depthStripsVal = emitConstI32(builder, depthStrips);

    const char *kernelName = "convDKKC8";
    auto *F = getFunction(kernelName, dest->getElementType());

    createCall(builder, F,
               {destPtr, srcPtr, filterPtr, biasPtr, destDims, srcDims,
                filterDims, biasDims, kernels, strides, pads, group,
                pixelScanFirstVal, numDepthRegsVal, sizeGroupYVal,
                depthStripsVal});
    break;
  }
case Kinded::Kind::ChannelwiseQuantizedConvolutionInstKind: {
    auto *CQCI = cast<ChannelwiseQuantizedConvolutionInst>(I);
    auto *dest = CQCI->getDest();
    auto *src = CQCI->getSrc();
    auto *filter = CQCI->getFilter();
    auto *bias = CQCI->getBias();
    auto *filterScales = CQCI->getFilterScales();
    auto *filterOffsets = CQCI->getFilterOffsets();
    auto *biasScales = CQCI->getBiasScales();
    auto *biasOffsets = CQCI->getBiasOffsets();

    auto *destTy = dest->getType();
    auto *srcTy = src->getType();

    auto filterScalesT = getTensorForConstantValue(filterScales);
    auto filterScalesH = filterScalesT.getHandle<float>();

    auto biasScalesT = getTensorForConstantValue(biasScales);
    auto biasScalesH = biasScalesT.getHandle<float>();

    // Compute quantization parameters for each channel.
    auto channelNum = dest->dims().back();
    std::vector<llvm::Constant *> biasPreV(channelNum);
    std::vector<llvm::Constant *> biasPostV(channelNum);
    std::vector<llvm::Constant *> biasScaleV(channelNum);
    std::vector<llvm::Constant *> outputPreV(channelNum);
    std::vector<llvm::Constant *> outputPostV(channelNum);
    std::vector<llvm::Constant *> outputScaleV(channelNum);

    std::vector<llvm::Constant *> cmsis_ScaleV(channelNum);
    std::vector<llvm::Constant *> cmsis_OffsetV(channelNum);
    for (size_t i = 0; i < channelNum; i++) {

      // Compute the scaling parameters for bias and output.
      float matMulScale = srcTy->getScale() * filterScalesH.raw(i);
      auto biasScaleParam = quantization::quantizeScaleOffset32To8(
          biasScalesH.raw(i) / matMulScale, 0);
      auto outScaleParam = quantization::quantizeScaleOffset32To8(
          matMulScale / destTy->getScale(), 0);
      auto cmsis_outScaleParam = quantization::CMSIS_quantizeScaleOffset32To8(
            matMulScale / destTy->getScale(), destTy->getOffset());

      // Pass the pre-shift, post-shift and integer scale parameters for the
      // bias and output calculation.
      biasPreV[i] = llvm::ConstantInt::get(builder.getInt32Ty(),
                                           biasScaleParam.pre, true);
      biasPostV[i] = llvm::ConstantInt::get(builder.getInt32Ty(),
                                            biasScaleParam.post, true);
      biasScaleV[i] = llvm::ConstantInt::get(builder.getInt32Ty(),
                                             biasScaleParam.scale, true);
      outputPreV[i] =
          llvm::ConstantInt::get(builder.getInt32Ty(), outScaleParam.pre, true);
      outputPostV[i] = llvm::ConstantInt::get(builder.getInt32Ty(),
                                              outScaleParam.post, true);
      outputScaleV[i] = llvm::ConstantInt::get(builder.getInt32Ty(),
                                               outScaleParam.scale, true);

      cmsis_ScaleV[i] = llvm::ConstantInt::get(builder.getInt32Ty(),
                                              cmsis_outScaleParam.cmsis_scale, true);
      cmsis_OffsetV[i] = llvm::ConstantInt::get(builder.getInt32Ty(),
                                               cmsis_outScaleParam.cmsis_offset, true);
    }

    auto *destPtr = emitValueAddress(builder, dest);
    auto *srcPtr = emitValueAddress(builder, src);
    auto *filterPtr = emitValueAddress(builder, filter);
    auto *biasPtr = emitValueAddress(builder, bias);

    auto *destDims = emitValueDims(builder, dest);
    auto *srcDims = emitValueDims(builder, src);
    auto *filterDims = emitValueDims(builder, filter);
    auto *biasDims = emitValueDims(builder, bias);

    auto *kernels = emitConstDimTArray(builder, CQCI->getKernels());
    auto *strides = emitConstDimTArray(builder, CQCI->getStrides());
    auto *pads = emitConstDimTArray(builder, CQCI->getPads());
    auto *group = emitConstDimT(builder, CQCI->getGroup());
    auto *dilation = emitConstDimTArray(builder, CQCI->getDilation());

    auto *destOffset = emitConstI32(builder, destTy->getOffset());
    auto *srcOffset = emitConstI32(builder, -(srcTy->getOffset()));
    auto *filterOffsetsPtr = emitValueAddress(builder, filterOffsets);
    auto *biasOffsetsPtr = emitValueAddress(builder, biasOffsets);

    auto *biasPrePtr = emitConstArray(builder, biasPreV, builder.getInt32Ty());
    auto *biasPostPtr =
        emitConstArray(builder, biasPostV, builder.getInt32Ty());
    auto *biasScalePtr =
        emitConstArray(builder, biasScaleV, builder.getInt32Ty());
    auto *outputPrePtr =
        emitConstArray(builder, outputPreV, builder.getInt32Ty());
    auto *outputPostPtr =
        emitConstArray(builder, outputPostV, builder.getInt32Ty());
    auto *outputScalePtr =
        emitConstArray(builder, outputScaleV, builder.getInt32Ty());

    auto *cmsis_ScalePtr =
        emitConstArray(builder, cmsis_ScaleV, builder.getInt32Ty());
    auto *cmsis_OffsetPtr =
        emitConstArray(builder, cmsis_OffsetV, builder.getInt32Ty());

    bool isDepthwise = (filter->dims()[3] == 1 && dest->dims()[3] == src->dims()[3]);
    auto *F = getFunction(isDepthwise ? "depthwise_conv2_3d_i8_i32_cmsis_wrapper"
                                      : "channelwise_conv2_3d_i8_i32_cmsis_wrapper");

    auto *actType = emitConstI32(builder, CQCI->getFusedActivation());
    auto *actArgsQuant = emitConstQuantActivationArgs(builder, CQCI);

    createCall(builder, F,
               {destPtr,        srcPtr,        filterPtr,      biasPtr,
                destDims,       srcDims,       filterDims,     biasDims,
                kernels,        strides,       pads,           group,
                dilation,       destOffset,    srcOffset,      filterOffsetsPtr,
                biasOffsetsPtr, biasPrePtr,    biasPostPtr,    biasScalePtr,
                outputPrePtr,   outputPostPtr, outputScalePtr, actType,
                actArgsQuant, cmsis_ScalePtr, cmsis_OffsetPtr });
    break;
  }
  case Kinded::Kind::FullyConnectedInstKind: {
      auto *FCI = cast<FullyConnectedInst>(I);
      auto *dest = FCI->getDest();
      auto *src = FCI->getSrc();
      auto *weights = FCI->getWeights();
      auto *bias = FCI->getBias();
      auto *destPtr = emitValueAddress(builder, dest);
      auto *srcPtr = emitValueAddress(builder, src);
      auto *weightsPtr = emitValueAddress(builder, weights);
      auto *biasPtr = emitValueAddress(builder, bias);
      auto *destDims = emitValueDims(builder, dest);
      auto *srcDims = emitValueDims(builder, src);
      auto *weightsDims = emitValueDims(builder, weights);
      auto *biasDims = emitValueDims(builder, bias);

      if (src->getType()->isQuantizedType()) {
        auto *destTy = dest->getType();
        auto *srcTy = src->getType();
        auto *weightsTy = weights->getType();
        auto *biasTy = bias->getType();

        auto *destOffset = emitConstI32(builder, destTy->getOffset());
        auto *srcOffset = emitConstI32(builder, -(srcTy->getOffset()));
        auto *weightsOffset = emitConstI32(builder, weightsTy->getOffset());
        auto *biasOffset = emitConstI32(builder, biasTy->getOffset());

        // Calculate the scale of the values that come out of the matrix
        // multiplication part of the calculation.
        float matMulScale = srcTy->getScale() * weightsTy->getScale();

        // Calculate the scaling parameters for the bias and output.
        auto biasScaleParam = quantization::quantizeScaleOffset32To8(
            biasTy->getScale() / matMulScale, 0);
        auto outScaleParam = quantization::quantizeScaleOffset32To8(
            matMulScale / destTy->getScale(), 0);
        auto cmsis_outScaleParam = quantization::CMSIS_quantizeScaleOffset32To8(
            matMulScale / destTy->getScale(), destTy->getOffset());

        // Pass the pre-shift, post-shift and integer scale parameters for the
        // bias and output calculation.
        auto *biasPre = emitConstI32(builder, biasScaleParam.pre);
        auto *biasPost = emitConstI32(builder, biasScaleParam.post);
        auto *biasScale = emitConstI32(builder, biasScaleParam.scale);
        auto *outPre = emitConstI32(builder, outScaleParam.pre);
        auto *outPost = emitConstI32(builder, outScaleParam.post);
        auto *outScale = emitConstI32(builder, outScaleParam.scale);
        auto *cmsis_scale = emitConstI32(builder, cmsis_outScaleParam.cmsis_scale);
        auto *cmsis_Offset = emitConstI32(builder, cmsis_outScaleParam.cmsis_offset);

        auto *F = getFunction("FC_s8_cmsis_wrapper");
        createCall(builder, F,
                  {destPtr, srcPtr, weightsPtr, biasPtr, destDims, srcDims,
                    weightsDims, biasDims, destOffset, srcOffset, weightsOffset,
                    biasOffset, biasPre, biasPost, biasScale, outPre, outPost,
                    outScale,cmsis_scale,cmsis_Offset});
      } else {
        assert("Glow CMSIS Backend does not supports FP Dense at the moment");
      }
  }
      break;
  default:
    LLVMIRGen::generateLLVMIRForInstr(builder, I);
  }
}

void CPULLVMIRGen::generateLLVMIRForDataParallelInstr(
    llvm::IRBuilder<> &builder, const glow::Instruction *I,
    llvm::Function *kernel, llvm::DenseMap<Value *, int> &bufferToArgNum,
    llvm::Value *loopCount) {
  setCurrentDebugLocation(builder, I);
  assert(canBePartOfDataParallelKernel(I) &&
         "Expected a data parallel instruction");
  // Perform any backend-specific code generation here and delegate everything
  // else to LLVMIRGen.
  switch (I->getKind()) {
  case Kinded::Kind::CPUMaxSplatInstKind: {
    auto *AN = cast<CPUMaxSplatInst>(I);
    auto *dest = AN->getDest();
    auto V = AN->getSplatValue();
    auto *destPtr = emitBufferAddress(builder, dest, kernel, bufferToArgNum);
    auto *lhs = AN->getSrc();
    auto *lhsPtr = emitBufferAddress(builder, lhs, kernel, bufferToArgNum);
    auto *F = getFunction("element_maxsplat_kernel", dest->getElementType());
    auto *elementTy = getElementType(builder, dest);
    auto *pointerNull =
        llvm::ConstantPointerNull::get(elementTy->getPointerTo());

    if (lhs->getType()->isQuantizedType()) {
      // Quantize value from the splat to the {S,O} of the lhs param.
      TensorQuantizationParams TQP{lhs->getType()->getScale(),
                                   lhs->getType()->getOffset()};
      auto quantizedValue = quantization::quantize(V, TQP);
      auto *val = emitConst(builder, quantizedValue, lhs->getElementType());
      auto *stackedOpCall = createUncheckedCall(
          builder, F, {loopCount, val, lhsPtr, pointerNull});
      auto *destAddr = builder.CreateGEP(builder.getInt8Ty(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    } else {
      auto *val = emitConst(builder, V, lhs->getElementType());
      auto *stackedOpCall = createUncheckedCall(
          builder, F, {loopCount, val, lhsPtr, pointerNull});
      auto *destAddr = builder.CreateGEP(builder.getFloatTy(), destPtr,
                                         loopCount, "buffer.element.addr");
      builder.CreateStore(stackedOpCall, destAddr);
    }

    break;
  }

  default:
    LLVMIRGen::generateLLVMIRForDataParallelInstr(builder, I, kernel,
                                                  bufferToArgNum, loopCount);
  }
}
