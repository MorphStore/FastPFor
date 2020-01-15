/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Daniel Lemire
 */
#ifndef SIMDBITPACKING256_H_
#define SIMDBITPACKING256_H_

#include "common.h"

namespace FastPForLib {

void simd256pack(const uint32_t *__restrict__ in, __m256i *__restrict__ out,
              uint32_t bit);
void simd256packwithoutmask(const uint32_t *__restrict__ in,
                         __m256i *__restrict__ out, uint32_t bit);
void simd256unpack(const __m256i *__restrict__ in, uint32_t *__restrict__ out,
                uint32_t bit);

void SIMD256_fastunpack_32(const __m256i *__restrict__ in,
                        uint32_t *__restrict__ out, const uint32_t bit);
void SIMD256_fastpackwithoutmask_32(const uint32_t *__restrict__ in,
                                 __m256i *__restrict__ out, const uint32_t bit);
void SIMD256_fastpack_32(const uint32_t *__restrict__ in,
                      __m256i *__restrict__ out, const uint32_t bit);

} // namespace FastPFor

#endif /* SIMDBITPACKING256_H_ */
