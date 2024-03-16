add_library(LLVMIRCodeGen
            AllocationsInfo.cpp
            BundleSaver.cpp
            CommandLine.cpp
            LLVMCompiledFunction.cpp
            DebugInfo.cpp
            JITFilePrinter.cpp
            FunctionSpecializer.cpp
            DebugInstrumentation.cpp
            GlowJIT.cpp
            Pipeline.cpp
            LLVMIRGen.cpp
            LLVMBackend.cpp)

llvm_map_components_to_libnames(LLVM_TARGET_LIBRARIES ${LLVM_TARGETS_TO_BUILD})
target_link_libraries(LLVMIRCodeGen
                      PUBLIC
                        Backend
                        Base
                        CodeGen
                        Flags
                        Graph
                        IR
                        IROptimizer
                        IROptimizerPipeline
                        GraphOptimizerPipeline
                        QuantizationBase
                        ${LLVM_TARGET_LIBRARIES}
                        LLVMAnalysis
                        LLVMBitWriter
                        LLVMCodeGen
                        LLVMCore
                        LLVMExecutionEngine
                        LLVMIRReader
                        LLVMInstCombine
                        LLVMInterpreter
                        LLVMMC
                        LLVMObject
                        LLVMPasses
                        LLVMScalarOpts
                        LLVMSupport
                        LLVMTarget
                        LLVMTransformUtils
                        LLVMVectorize
                        LLVMipo)
if(LLVM_VERSION_MAJOR VERSION_GREATER 6)
  target_link_libraries(LLVMIRCodeGen
                        PRIVATE
                          LLVMOrcJIT)
endif()

if (GLOW_WITH_CMSIS)
    set(CMSIS_NN_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c
	${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c
	${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c
	${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8.c
	${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c
	${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c
	${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c
	${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
	${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c
	${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.c
	${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.c
	${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c
	${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c
	${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.c
	${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s8.c
	${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.c
	${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c
	${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c
   )

    add_library(cmsis-nn STATIC ${CMSIS_NN_SOURCES})
    target_include_directories(cmsis-nn PUBLIC "CMSIS-NN/Include")
    target_link_libraries(LLVMIRCodeGen
    		          PRIVATE
		            cmsis-nn)
endif()
