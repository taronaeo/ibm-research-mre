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
 * This file is a template for vector extensions (VXE) implementation.
 * It includes definitions for vector types and operations for both s390x and ARM NEON architectures.
 * The code is designed to be compiled with GCC or Clang.
 *
 * Build binary using GCC
 * `gcc -O3 -march=z15 -mvx -mzvector vxe_template.c -o vectest`
 *
 * Build assembly using GCC
 * `gcc -S -O3 -march=z15 -mvx -mzvector vxe_template.c -o vectest`
 */

#include <string.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __s390x__
#include <vecintrin.h>
#define vec_neg(a)    (-(a))                // Vector Negate
#define vec_add(a, b) ((a) + (b))           // Vector Add
#define vec_sub(a, b) ((a) - (b))           // Vector Subtract
#define vec_mul(a, b) ((a) * (b))           // Vector Multiply
#define vec_div(a, b) ((a) / (b))           // Vector Divide
#define  vec_sl(a, b) ((a) << (b))          // Vector Shift Left
#define vec_sra(a, b) ((a) >> (b))          // Vector Shift Right
#define  vec_sr(a, b) ((a) >> (b))          // Vector Shift Right Algebraic
#define vec_slo(a, b) vec_slb(a, (b) << 64) // Vector Shift Left by Octet
#define vec_sro(a, b) vec_srb(a, (b) << 64) // Vector Shift Right by Octet

#ifndef vec_and
#define vec_and(a, b) ((a) & (b)) // Vector AND
#endif

#ifndef vec_or
#define vec_or(a, b)  ((a) | (b)) // Vector OR
#endif

#ifndef vec_xor
#define vec_xor(a, b) ((a) ^ (b)) // Vector XOR
#endif

typedef signed   char char8x16_t  __attribute__((vector_size(16)));
typedef unsigned char uchar8x16_t __attribute__((vector_size(16)));

typedef int8_t   int8x16_t  __attribute__((vector_size(16)));
typedef int16_t  int16x8_t  __attribute__((vector_size(16)));
typedef int32_t  int32x4_t  __attribute__((vector_size(16)));
typedef uint8_t  uint8x16_t __attribute__((vector_size(16)));
typedef uint16_t uint16x8_t __attribute__((vector_size(16)));
typedef uint32_t uint32x4_t __attribute__((vector_size(16)));

typedef float  float32x4_t  __attribute__((vector_size(16)));
typedef double double64x2_t __attribute__((vector_size(16)));

typedef signed   long long long64x2_t  __attribute__((vector_size(16)));
typedef unsigned long long ulong64x2_t __attribute__((vector_size(16)));

/*
 * Example implementation of a horizontal sum function to compare against ARM64
 */
float __attribute__((noinline)) hsum(float32x4_t x) {}

#endif

#ifdef __ARM_NEON
#include <arm_neon.h>

/*
 * Example implementation of a horizontal sum function to compare against s390x
 */
float __attribute__((noinline)) hsum(float32x4_t x) {}

#endif

void printv_u8(uint8x16_t in) {
    for (int i = 0; i < 16; i++) {
        printf("%s: %d ", __func__, in[i]);
    }
    printf("\n");
}

void printv_s8(int8x16_t in) {
    for (int i = 0; i < 16; i++) {
        printf("%s: %d ", __func__, in[i]);
    }
    printf("\n");
}

void printv_u16(uint16x8_t in) {
    for (int i = 0; i < 8; i++) {
        printf("%s: %d ", __func__, in[i]);
    }
    printf("\n");
}

void printv_s16(int16x8_t in) {
    for (int i = 0; i < 8; i++) {
        printf("%s: %d ", __func__, in[i]);
    }
    printf("\n");
}

void printv_u32(uint32x4_t in) {
    for (int i = 0; i < 4; i++) {
        printf("%s: %d ", __func__, in[i]);
    }
    printf("\n");
}

void printv_s32(int32x4_t in) {
    for (int i = 0; i < 4; i++) {
        printf("%s: %d ", __func__, in[i]);
    }
    printf("\n");
}

void printv_f32(float32x4_t in) {
    for (int i = 0; i < 4; i++) {
        printf("%s: %f ", __func__, in[i]);
    }
    printf("\n");
}

int main() {
    static const float32x4_t test = { 1, 2, 3, 4 };
    float result = hsum(test);

    return 0;
}
