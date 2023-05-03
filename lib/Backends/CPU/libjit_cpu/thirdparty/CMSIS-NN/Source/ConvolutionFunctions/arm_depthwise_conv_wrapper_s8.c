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
 * Title:        arm_depthwise_conv_wrapper_s8.c
 * Description:  Wrapper API to select appropriate depthwise conv API based
 *               on dimensions.
 *
 * $Date:        13 January 2023
 * $Revision:    V.2.1.0
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
 *  s8 Depthwise conv wrapper function
 *
 *  Refer header file for details.
 *
 */
arm_cmsis_nn_status arm_depthwise_conv_wrapper_s8(const cmsis_nn_context *ctx,
                                                  const cmsis_nn_dw_conv_params *dw_conv_params,
                                                  const cmsis_nn_per_channel_quant_params *quant_params,
                                                  const cmsis_nn_dims *input_dims,
                                                  const int8_t *input,
                                                  const cmsis_nn_dims *filter_dims,
                                                  const int8_t *filter,
                                                  const cmsis_nn_dims *bias_dims,
                                                  const int32_t *bias,
                                                  const cmsis_nn_dims *output_dims,
                                                  int8_t *output)
{
    // printf("filter_data[0] == %d\n",filter[0]);

    // printf("quant_params_multiplier[0] == %d\n",quant_params->multiplier[0]);
    // printf("quant_params_shift[0] == %d\n",quant_params->shift[0]);
    
    // printf("conv_params->input_offset == %d\n",dw_conv_params->input_offset);
    // printf("conv_params->output_offset == %d\n", dw_conv_params->output_offset);
    // printf("conv_params->activation.min %d\n",dw_conv_params->activation.min);
    // printf("conv_params->activation.max %d\n",dw_conv_params->activation.max);
    // printf("conv_params->ch_mult %d\n",dw_conv_params->ch_mult);
    // printf("filter_dims->n == %d\n",filter_dims->n);
    // printf(" output_dims->c == %d\n",output_dims->c);

    // printf("input_dims->w %d\n",input_dims->w);
    // printf("input_dims->h %d\n",input_dims->h);
    // printf("input_dims->c %d\n",input_dims->c);
    // printf("filter_dims->w %d\n",filter_dims->w);
    // printf("filter_dims->h %d\n",filter_dims->h);
    // printf("output_dims->w %d\n",output_dims->w);
    // printf("output_dims->h %d\n",output_dims->h);
    // printf("output_dims->c %d\n",output_dims->c);

    arm_cmsis_nn_status status = ARM_CMSIS_NN_SUCCESS;
    if (1 == dw_conv_params->ch_mult && input_dims->n == 1 && dw_conv_params->dilation.w == 1 &&
        dw_conv_params->dilation.h == 1)
    {
#if !defined(ARM_MATH_MVEI)
        if (filter_dims->w == 3 && filter_dims->h == 3 && dw_conv_params->padding.h <= 1 &&
            dw_conv_params->padding.w <= 1)
        {
            status = arm_depthwise_conv_3x3_s8(ctx,
                                               dw_conv_params,
                                               quant_params,
                                               input_dims,
                                               input,
                                               filter_dims,
                                               filter,
                                               bias_dims,
                                               bias,
                                               output_dims,
                                               output);
        }
        else
#endif
        {
            status = arm_depthwise_conv_s8_opt(ctx,
                                               dw_conv_params,
                                               quant_params,
                                               input_dims,
                                               input,
                                               filter_dims,
                                               filter,
                                               bias_dims,
                                               bias,
                                               output_dims,
                                               output);
        }
    }
    else
    {
        status = arm_depthwise_conv_s8(ctx,
                                       dw_conv_params,
                                       quant_params,
                                       input_dims,
                                       input,
                                       filter_dims,
                                       filter,
                                       bias_dims,
                                       bias,
                                       output_dims,
                                       output);
    }

    /* Return to application */
    return status;
}

/**
 * @} end of NNConv group
 */
