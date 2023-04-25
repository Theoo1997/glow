#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <iostream>
#include <vector>

#include "../../../LLVMIRCodeGen/libjit/libjit_defs.h"


#include "thirdparty/CMSIS-NN/Include/arm_nnfunctions.h"
#include "thirdparty/CMSIS-NN/Include/arm_nn_math_types.h"
#include "thirdparty/CMSIS-NN/Include/arm_nnsupportfunctions.h"

extern "C" {
void libjit_FC_s8_cmsis_wrapper(int8_t *outW, const int8_t *inW, const int8_t *weightsW,
                      const int32_t *biasW, const dim_t *outWdims,
                      const dim_t *inWdims, const dim_t *weightsWdims,
                      const dim_t *biasWdims, int32_t outOffset,
                      int32_t inOffset, int32_t weightsOffset,
                      int32_t biasOffset, int32_t biasPre, int32_t biasPost, 
                      int32_t biasScale, int32_t outPre, int32_t outPost,
                      int32_t outScale, int32_t cmsis_scale,int32_t cmsis_Offset) {
    
    (void)outPost,(void)outPre,(void)biasScale,(void)biasPost,(void)biasPre,(void)biasOffset,(void)biasWdims,(void)outWdims; // Not used

    cmsis_nn_context context= {NULL,0}; // Not used 
    cmsis_nn_activation activation = {-128,127}; // No activation used
    cmsis_nn_fc_params fc_params = {inOffset, weightsOffset, outOffset, activation};
    cmsis_nn_per_tensor_quant_params quant_params = { cmsis_scale, cmsis_Offset};
    cmsis_nn_dims input_dims = {(int32_t)  inWdims[0],1,1, (int32_t) inWdims[1]};
    cmsis_nn_dims filter_dims = {(int32_t) weightsWdims[0],1,1, (int32_t) weightsWdims[1]};
    cmsis_nn_dims bias_dims = {1,1,1,(int32_t) weightsWdims[1]};
    cmsis_nn_dims output_dims = {(int32_t) inWdims[0], 1, 1, (int32_t) weightsWdims[1]};

    // Reorder weightsW
    // Glow load filter weights in a different layout (weightsWdims[1] x inWdims[1]) than models are represented (inWdims[1] x weightsWdims[1]).
    // So reorder must be applied in order to have correct output.
    int8_t *weightsW_reord = nullptr;
    libjit_aligned_malloc((void **) &weightsW_reord,8 ,weightsWdims[1]*inWdims[1]);
    for (int i = 0; i < weightsWdims[1]; i++){
        for (int j = 0; j < inWdims[1]; j++){
            weightsW_reord[i*inWdims[1] + j] = weightsW[j*weightsWdims[1] + i];
        }
    }

    arm_cmsis_nn_status status = arm_fully_connected_s8(&context, &fc_params, &quant_params, &input_dims, inW, &filter_dims, weightsW_reord, &bias_dims, biasW, &output_dims, outW);

    if (status != ARM_CMSIS_NN_SUCCESS) { // TOODO: change this check.
        printf("ERROR in CMSIS arm_fully_connected_s8\n");
    }
    libjit_aligned_free(weightsW_reord);                                                       
}

void libjit_depthwise_conv2_3d_i8_i32_cmsis_wrapper(
    int8_t *outW, const int8_t *inW, const int8_t *filterW,
    const int32_t *biasW, const dim_t *outWdims, const dim_t *inWdims,
    const dim_t *filterWdims, const dim_t *biasWdims, const dim_t *kernels,
    const dim_t *strides, const dim_t *pads, dim_t group, const dim_t *dilation,
    int32_t outOffset, int32_t inOffset, int32_t *filterOffsetsPtr,
    int32_t *biasOffsetsPtr, const int32_t *biasPrePtr,
    const int32_t *biasPostPtr, const int32_t *biasScalePtr,
    const int32_t *outPrePtr, const int32_t *outPostPtr,
    const int32_t *outScalePtr, int32_t actType, const int32_t *actArgs,const int32_t * cmsis_ScaleV,const int32_t * cmsis_OffsetV) {

    // TODOO: This initilizations msut not be done here!! I will implemented before this step. 
    arm_cmsis_nn_status status;
    cmsis_nn_activation activation;
    
    cmsis_nn_dims input_dims = {(int32_t) inWdims[0],(int32_t) inWdims[2],(int32_t) inWdims[1], (int32_t) inWdims[3]};
    cmsis_nn_dims filter_dims = {(int32_t)  outWdims[3],(int32_t) kernels[0],(int32_t) kernels[0], (int32_t) inWdims[3]};
    cmsis_nn_dims bias_dims = {(int32_t) inWdims[0],(int32_t) biasWdims[1],(int32_t) biasWdims[2], (int32_t) outWdims[3]};
    cmsis_nn_dims output_dims = {(int32_t) inWdims[0],(int32_t) outWdims[1],(int32_t) outWdims[2], (int32_t) outWdims[3]};

    cmsis_nn_per_channel_quant_params quant_params = {(int32_t *) cmsis_ScaleV,(int32_t *) cmsis_OffsetV}; // TOODO: change this with the write scale and offset
    cmsis_nn_tile stride = {(int32_t) strides[1],(int32_t) strides[0]};
    cmsis_nn_tile padding = {(int32_t) pads[1],(int32_t) pads[0]};
    cmsis_nn_tile CMSIS_dilation = {(int32_t) dilation[1],(int32_t) dilation[0]};

    // TODOO: Add support for clip and lRELU, and make sure this is correct!! 
    if (actType == 0) {
    // No activation.
    activation = {-128,127};
    }
    else { // (actType == 1)  } 
    // Relu.
    activation = {0,outOffset};
    }
    cmsis_nn_dw_conv_params conv_params = {inOffset,/*ch_mult == */ 1, outOffset, stride, padding, CMSIS_dilation, activation}; // TODOO: find cases where ch_mult != 1 
    
    //Alocate Scratch Buffer for cmsis. TODOO Put it when libjit is builded.
    float *cmsis_buffer = nullptr;
    int32_t  buffer = arm_depthwise_conv_wrapper_s8_get_buffer_size(&conv_params,&input_dims, &filter_dims, &output_dims);
                                                                                                           
    libjit_aligned_malloc((void **) &cmsis_buffer,64 ,buffer);
    cmsis_nn_context ctx = {(void*) cmsis_buffer,0};
    
    status = arm_depthwise_conv_wrapper_s8(&ctx,&conv_params,&quant_params,&input_dims,inW,&filter_dims,filterW,&bias_dims,biasW,&output_dims,outW);
    if (status != ARM_CMSIS_NN_SUCCESS){
      status ==  ARM_CMSIS_NN_ARG_ERROR ? printf("Scratch Buffer error\n") : printf("ARM_CMSIS_NN_NO_IMPL_ERROR\n");
    }
    libjit_aligned_free(cmsis_buffer);
}

void libjit_channelwise_conv2_3d_i8_i32_cmsis_wrapper(
    int8_t *outW, const int8_t *inW, const int8_t *filterW,
    const int32_t *biasW, const dim_t *outWdims, const dim_t *inWdims,
    const dim_t *filterWdims, const dim_t *biasWdims, const dim_t *kernels,
    const dim_t *strides, const dim_t *pads, dim_t group, const dim_t *dilation,
    int32_t outOffset, int32_t inOffset, int32_t *filterOffsetsPtr,
    int32_t *biasOffsetsPtr, const int32_t *biasPrePtr,
    const int32_t *biasPostPtr, const int32_t *biasScalePtr,
    const int32_t *outPrePtr, const int32_t *outPostPtr,
    const int32_t *outScalePtr, int32_t actType, const int32_t *actArgs,const int32_t * cmsis_ScaleV,const int32_t * cmsis_OffsetV) {

    // TODOO: This initilizations msut not be done here!! I will implemented before this step. 
    arm_cmsis_nn_status status;
    cmsis_nn_activation activation;
    
    cmsis_nn_dims input_dims = {(int32_t) inWdims[0],(int32_t) inWdims[2],(int32_t) inWdims[1], (int32_t) inWdims[3]};
    cmsis_nn_dims filter_dims = {(int32_t)  outWdims[3],(int32_t) kernels[0],(int32_t) kernels[0], (int32_t) inWdims[3]};
    cmsis_nn_dims bias_dims = {(int32_t) inWdims[0],(int32_t) biasWdims[1],(int32_t) biasWdims[2], (int32_t) outWdims[3]};
    cmsis_nn_dims output_dims = {(int32_t) inWdims[0],(int32_t) outWdims[1],(int32_t) outWdims[2], (int32_t) outWdims[3]};

    cmsis_nn_per_channel_quant_params quant_params = {(int32_t *) cmsis_ScaleV,(int32_t *) cmsis_OffsetV}; // TOODO: change this with the write scale and offset
    cmsis_nn_tile stride = {(int32_t) strides[1],(int32_t) strides[0]};
    cmsis_nn_tile padding = {(int32_t) pads[1],(int32_t) pads[0]};
    cmsis_nn_tile CMSIS_dilation = {(int32_t) dilation[1],(int32_t) dilation[0]};

    // TODOO: Add support for clip and lRELU, and make sure this is correct!! 
    if (actType == 0) {
    // No activation.
    activation = {-128,127};
    }
    else { // (actType == 1)  } 
    // Relu.
    activation = {0,outOffset};
    }
    cmsis_nn_conv_params conv_params = {inOffset, outOffset, stride, padding, CMSIS_dilation, activation};
    
    //Alocate Scratch Buffer for cmsis. TODOO Put it when libjit is builded.
    float *cmsis_buffer = nullptr;
    int32_t buffer = arm_convolve_wrapper_s8_get_buffer_size(&conv_params,&input_dims, &filter_dims, &output_dims);
    libjit_aligned_malloc((void **) &cmsis_buffer,64 ,buffer);
    cmsis_nn_context ctx = {(void*) cmsis_buffer,0};
    // TODOO. This is not good implimeteantion. 
    //This wrapper should be done on CPULLVMIRGen. 
    //Calling a wrapper from wrapper will put unessesery implinations of Kernels in libjit that are not used (unessesery RAM usage from .o). 
    status = arm_convolve_wrapper_s8(&ctx,&conv_params,&quant_params,&input_dims,inW,&filter_dims,filterW,&bias_dims,biasW,&output_dims,outW); 
    if (status != ARM_CMSIS_NN_SUCCESS){
      status ==  ARM_CMSIS_NN_ARG_ERROR ? printf("Scratch Buffer error\n") : printf("ARM_CMSIS_NN_NO_IMPL_ERROR\n");
    }
    libjit_aligned_free(cmsis_buffer);
}

}// extern "C"