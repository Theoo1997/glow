/*
 * SPDX-FileCopyrightText: Copyright 2010-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_convolve_wrapper_s8.c
 * Description:  s8 convolution layer wrapper function with the main purpose to call the optimal kernel available in
 * cmsis-nn to perform the convolution.
 *
 * $Date:        11 January 2023
 * $Revision:    V.2.3.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */
#include <stdio.h>
#include "../../Include/arm_nnfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup NNConv
 * @{
 */

/*
 * Convolution layer
 *
 * Refer header file for details.
 *
 */

arm_cmsis_nn_status arm_convolve_wrapper_s8(const cmsis_nn_context *ctx,
                                            const cmsis_nn_conv_params *conv_params,
                                            const cmsis_nn_per_channel_quant_params *quant_params,
                                            const cmsis_nn_dims *input_dims,
                                            const int8_t *input_data,
                                            const cmsis_nn_dims *filter_dims,
                                            const int8_t *filter_data,
                                            const cmsis_nn_dims *bias_dims,
                                            const int32_t *bias_data,
                                            const cmsis_nn_dims *output_dims,
                                            int8_t *output_data)
{
    /*printf("filter_data[0] == %d\n",filter_data[0]);
    printf("filter_data[26] == %d\n",filter_data[26]);

    printf("quant_params_multiplier[0] == %d\n",quant_params->multiplier[0]);
    printf("quant_params_shift[0] == %d\n",quant_params->shift[0]);

    printf("quant_params_multiplier[2] == %d\n",quant_params->multiplier[2]);
    printf("quant_params_shift[2] == %d\n",quant_params->shift[2]);
    
    printf("conv_params->input_offset == %d\n",conv_params->input_offset);
    printf("conv_params->output_offset == %d\n", conv_params->output_offset);
    printf("filter_dims->n == %d\n",filter_dims->n);
    printf(" output_dims->c == %d\n",output_dims->c);
    printf("conv_params->activation.min %d\n",conv_params->activation.min);
    printf("conv_params->activation.max %d\n",conv_params->activation.max);

    printf("input_dims->w %d\n",input_dims->w);
    printf("input_dims->h %d\n",input_dims->h);
    printf("input_dims->c %d\n",input_dims->c);
    printf("filter_dims->w %d\n",filter_dims->w);
    printf("filter_dims->h %d\n",filter_dims->h);
    printf("output_dims->w %d\n",output_dims->w);
    printf("output_dims->h %d\n",output_dims->h);
    printf("output_dims->c %d\n",output_dims->c);*/
    //printf("\n\n");

    if ((conv_params->padding.w == 0) && (conv_params->padding.h == 0) && (filter_dims->w == 1) &&
        (filter_dims->h == 1) && (conv_params->dilation.w == 1 && conv_params->dilation.h == 1))
    {
        if ((conv_params->stride.w == 1) && (conv_params->stride.h == 1))
        {
            return arm_convolve_1x1_s8_fast(ctx,
                                            conv_params,
                                            quant_params,
                                            input_dims,
                                            input_data,
                                            filter_dims,
                                            filter_data,
                                            bias_dims,
                                            bias_data,
                                            output_dims,
                                            output_data);
        }
        else
        {
            return arm_convolve_1x1_s8(ctx,
                                       conv_params,
                                       quant_params,
                                       input_dims,
                                       input_data,
                                       filter_dims,
                                       filter_data,
                                       bias_dims,
                                       bias_data,
                                       output_dims,
                                       output_data);
        }
    }
    else if ((input_dims->h == 1) && (output_dims->w % 4 == 0) && conv_params->dilation.w == 1 && (filter_dims->h == 1))
    {
        return arm_convolve_1_x_n_s8(ctx,
                                     conv_params,
                                     quant_params,
                                     input_dims,
                                     input_data,
                                     filter_dims,
                                     filter_data,
                                     bias_dims,
                                     bias_data,
                                     output_dims,
                                     output_data);
    }
    else
    {
         arm_cmsis_nn_status status = arm_convolve_s8(ctx,
                               conv_params,
                               quant_params,
                               input_dims,
                               input_data,
                               filter_dims,
                               filter_data,
                               bias_dims,
                               bias_data,
                               output_dims,
                               output_data);
        //printf("output_data[0] %d\n",output_data[0]);
        //printf("output_data[2] %d\n",output_data[2]);
        //printf("output_data[10] %d\n",output_data[10]);
        return status;
    }
}

/**
 * @} end of NNConv group
 */
