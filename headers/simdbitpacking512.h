/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Daniel Lemire
 */
#ifndef SIMDBITPACKING512_H_
#define SIMDBITPACKING512_H_

#include "common.h"

namespace FastPForLib {

void simd512pack(const uint32_t *__restrict__ in, __m512i *__restrict__ out,
              uint32_t bit);
void simd512packwithoutmask(const uint32_t *__restrict__ in,
                         __m512i *__restrict__ out, uint32_t bit);
void simd512unpack(const __m512i *__restrict__ in, uint32_t *__restrict__ out,
                uint32_t bit);

void SIMD512_fastunpack_32(const __m512i *__restrict__ in,
                        uint32_t *__restrict__ out, const uint32_t bit);
void SIMD512_fastpackwithoutmask_32(const uint32_t *__restrict__ in,
                                 __m512i *__restrict__ out, const uint32_t bit);
void SIMD512_fastpack_32(const uint32_t *__restrict__ in,
                      __m512i *__restrict__ out, const uint32_t bit);

} // namespace FastPFor

#endif /* SIMDBITPACKING512_H_ */
