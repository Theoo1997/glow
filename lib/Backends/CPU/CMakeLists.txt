add_subdirectory(ClassGen)

include(GlowAddLibjit)
include(GlowSerialize)

# LIBJIT LLVM common source files.
set(LIBJIT_CPU_SOURCE_FILES)
list(APPEND LIBJIT_CPU_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../LLVMIRCodeGen/libjit/libjit.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../LLVMIRCodeGen/libjit/libjit_conv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../LLVMIRCodeGen/libjit/libjit_matmul.cpp
)

# LIBJIT CPU specific source files. TODOO. Use other cmake for this
list(APPEND LIBJIT_CPU_SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/libjit_cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/libjit_cpu_conv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/libjit_CMSIS.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8.c
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.c
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.c
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.c
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s8.c
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.c
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c
  ${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c
)

# LIBJIT CPU compile options.
set(LIBJIT_CPU_COMPILE_OPTIONS
  -I${CMAKE_CURRENT_SOURCE_DIR}/../../LLVMIRCodeGen/libjit
  -I${CMAKE_CURRENT_SOURCE_DIR}/libjit_cpu/thirdparty/CMSIS-NN/Include
  -std=c++14
  -g
  -O0

  # When building for a custom target, use this as an example of how to
  # set up cross-compilation correctly. Then, when building bundles,
  # do not forget to specify the -target= option with the same target
  # string. Also, when building bundles, specify the -llvm-compiler option
  # together with -llvm-compiler-opt to specify custom compilation flags for
  # cross-compilation.

  # --sysroot=/usr/arm-linux-gnueabihf/
  # -target armv7-neon-linux-gnueabihf
  # -I/usr/arm-linux-gnueabihf/include/c++/7.4.0/arm-linux-gnueabihf/
  ${LLVMCPURuntimeExtraFlags}

  # Include LIBJIT external functions (callbacks).
  -DGLOW_LIBJIT_EXTERNAL_FUNCTIONS
)

# Add CPU LIBJIT.
glow_add_libjit(
  NAME "libjit_cpu"
  SOURCE_FILES ${LIBJIT_CPU_SOURCE_FILES}
  COMPILE_OPTIONS ${LIBJIT_CPU_COMPILE_OPTIONS}
)

# Add native CPU LIBJIT library used for testing.
if (NOT MSVC)
  add_library(CPURuntimeNative
              ${LIBJIT_CPU_SOURCE_FILES})
endif(NOT MSVC)

add_library(CPUBackend
            ${libjit_cpu_INCLUDE_FILE}
            CPUBackend.cpp
            CPUDeviceManager.cpp
            CPUFactory.cpp
            CPUFunction.cpp
            CPULLVMIRGen.cpp
            Transforms.cpp)

target_link_libraries(CPUBackend
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
                        Runtime
                        LLVMIRCodeGen)

add_dependencies(CPUBackend libjit_cpu_TARGET)

set(linked_backends ${linked_backends} CPUBackend PARENT_SCOPE)

# List the object files from "libobj_cpu" directory.
FILE(GLOB objects "${CMAKE_CURRENT_LIST_DIR}/libobj_cpu/*.o")

# Add object file for testing.
if (GLOW_WITH_BUNDLES)
  list(APPEND objects "${GLOW_SOURCE_DIR}/examples/bundles/bundle_with_extra_objects/test.o")
endif()

# Serialize external object files.
glow_serialize(
  INP_FILES ${objects}
  OUT_DIR "${GLOW_BINARY_DIR}/libobj/libobj_cpu"
  OUT_FILE "${GLOW_BINARY_DIR}/libobj/libobj_cpu/cpuObjectRegistry.h"
  OUT_VAR "cpuObjectRegistry"
  OUT_TARGET cpuObjectRegistryTarget
)
add_dependencies(CPUBackend cpuObjectRegistryTarget)
include_directories(${GLOW_BINARY_DIR}/libobj/libobj_cpu)
