/*
 * MIT License
 *
 * Copyright (c) 2025 Aaron Teo
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*
 * This file is a template for zDNN (z/Architecture Deep Neural Network) implementation.
 *
 * Build binary using GCC
 * `gcc -O3 -march=z16 -mvx -mzvector -lzdnn -I/opt/zdnn-libs/include -L/opt/zdnn-libs/lib zdnn_template.c -o zdnn`
 *
 * Build assembly using GCC
 * `gcc -S -O3 -march=z16 -mvx -mzvector -lzdnn -I/opt/zdnn-libs/include -L/opt/zdnn-libs/lib zdnn_template.c -o zdnn`
 */

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>

#include "zdnn.h"

int main(int argc, char *argv[]) {
#ifdef STATIC_LIB
    zdnn_init();
#endif

    zdnn_tensor_desc input1_pre_tfm_desc, input1_tfm_desc;
    zdnn_tensor_desc input2_pre_tfm_desc, input2_tfm_desc;
    zdnn_tensor_desc output_pre_tfm_desc, output_tfm_desc;

    zdnn_ztensor z_input1_tensor, z_input2_tensor, z_output_tensor;
    zdnn_status status;

    printf("%s: Is zDNN NNPA installed? %d\n", __func__, zdnn_is_nnpa_installed());
    printf("%s: Number of elements in any dimension must not exceed: %d\n", __func__, zdnn_get_nnpa_max_dim_idx_size());
    printf("%s: Total number of bytes required for storing a transformed tensor must not exceed: %" PRIu64 "\n", __func__, zdnn_get_nnpa_max_tensor_size());

    // Step 1: Specify NCHW dimensions and data type
    uint32_t input1_dim_n = 1, input1_dim_c = 1, input1_dim_h = 1, input1_dim_w = 2048;
    uint32_t input2_dim_n = 1, input2_dim_c = 1, input2_dim_h = 1, input2_dim_w = 2048;
    uint32_t output_dim_n = 1, output_dim_c = 1, output_dim_h = 1, output_dim_w = 2048;
    zdnn_data_types type = FP32;

    uint64_t input1_num_elements = input1_dim_n * input1_dim_c * input1_dim_h * input1_dim_w;
    uint64_t input2_num_elements = input2_dim_n * input2_dim_c * input2_dim_h * input2_dim_w;
    uint64_t output_num_elements = output_dim_n * output_dim_c * output_dim_h * output_dim_w;

    // Step 2: Correlate the `sizeof(float)` against your chosen data type
    uint64_t input1_buffer_size = input1_num_elements * sizeof(float);
    uint64_t input2_buffer_size = input2_num_elements * sizeof(float);
    uint64_t output_buffer_size = output_num_elements * sizeof(float);

    void * input1_data = malloc(input1_buffer_size);
    void * input2_data = malloc(input2_buffer_size);
    void * output_data = malloc(output_buffer_size);

    printf("%s: initialising input data...\n", __func__);
    // Step 3: Correlate the `(float *)` against your chosen data type
    for (uint64_t i = 0; i < input1_num_elements; i++) ((float *)input1_data)[i] = (float)(i & 0x7f) + 1.0f;
    for (uint64_t i = 0; i < input2_num_elements; i++) ((float *)input2_data)[i] = (float)(i & 0x7f) + 2.0f;
    printf("%s: initialised input1_data length: %zu\n", __func__, input1_num_elements);
    printf("%s: initialised input2_data length: %zu\n", __func__, input2_num_elements);

    zdnn_init_pre_transformed_desc(ZDNN_NCHW,
                                   type,
                                   &input1_pre_tfm_desc,
                                   input1_dim_n, input1_dim_c, input1_dim_h, input1_dim_w);
    zdnn_init_pre_transformed_desc(ZDNN_NCHW,
                                   type,
                                   &input2_pre_tfm_desc,
                                   input2_dim_n, input2_dim_c, input2_dim_h, input2_dim_w);
    zdnn_init_pre_transformed_desc(ZDNN_NCHW,
                                   type,
                                   &output_pre_tfm_desc,
                                   output_dim_n, output_dim_c, output_dim_h, output_dim_w);

    status = zdnn_generate_transformed_desc(&input1_pre_tfm_desc, &input1_tfm_desc); assert(status == ZDNN_OK);
    status = zdnn_generate_transformed_desc(&input2_pre_tfm_desc, &input2_tfm_desc); assert(status == ZDNN_OK);
    status = zdnn_generate_transformed_desc(&output_pre_tfm_desc, &output_tfm_desc); assert(status == ZDNN_OK);

    status = zdnn_init_ztensor_with_malloc(&input1_pre_tfm_desc, &input1_tfm_desc, &z_input1_tensor); assert(status == ZDNN_OK);
    status = zdnn_init_ztensor_with_malloc(&input2_pre_tfm_desc, &input2_tfm_desc, &z_input2_tensor); assert(status == ZDNN_OK);
    status = zdnn_init_ztensor_with_malloc(&output_pre_tfm_desc, &output_tfm_desc, &z_output_tensor); assert(status == ZDNN_OK);

    printf("%s: transforming input tensors into ztensor...\n", __func__);
    status = zdnn_transform_ztensor(&z_input1_tensor, input1_data); assert(status == ZDNN_OK);
    status = zdnn_transform_ztensor(&z_input2_tensor, input2_data); assert(status == ZDNN_OK);

    printf("%s: performing zdnn_mul ops...\n", __func__);
    status = zdnn_mul(&z_input1_tensor, &z_input2_tensor, &z_output_tensor);
    assert(status == ZDNN_OK);
    printf("%s: zdnn_mul operation completed successfully.\n", __func__);

    printf("%s: transforming result ztensor back to original format...\n", __func__);
    status = zdnn_transform_origtensor(&z_output_tensor, output_data);
    assert(status == ZDNN_OK);
    printf("%s: transformed ztensor back to original format successfully.\n", __func__);

    printf("--- verifying results ---\n");
    int failed_tests = 0;
    const float tolerance = 1e-6f;

    for (uint64_t i = 0; i < output_num_elements; i++) {
        float input1 = ((float *)input1_data)[i];
        float input2 = ((float *)input2_data)[i];
        float expected_result = input1 * input2;
        float actual_result = ((float *)output_data)[i];

        printf("Index %4" PRIu64 ": %5.2f * %5.2f | Expected %8.2f | Got %8.2f | ",
               i, input1, input2, expected_result, actual_result);

        if (fabs(expected_result - actual_result) > tolerance) {
            printf("FAILED\n");
            failed_tests++;
        } else {
            printf("PASSED\n");
        }
    }
    printf("--- verification summary ---\n");
    if (failed_tests == 0) {
        printf("all %" PRIu64 " test cases passed!\n", output_num_elements);
    } else {
        printf("%d out of %" PRIu64 " test cases failed.\n", failed_tests, output_num_elements);
    }
    printf("----------------------------\n");

    status = zdnn_free_ztensor_buffer(&z_input1_tensor); assert(status == ZDNN_OK);
    status = zdnn_free_ztensor_buffer(&z_input2_tensor); assert(status == ZDNN_OK);
    status = zdnn_free_ztensor_buffer(&z_output_tensor); assert(status == ZDNN_OK);

    free(input1_data);
    free(input2_data);
    free(output_data);

    return 0;
}
