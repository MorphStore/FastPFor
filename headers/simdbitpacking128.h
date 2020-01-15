/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Daniel Lemire
 */
#ifndef SIMDBITPACKING128_H_
#define SIMDBITPACKING128_H_

#include "common.h"

namespace FastPForLib {

void simd128pack(const uint32_t *__restrict__ in, __m128i *__restrict__ out,
              uint32_t bit);
void simd128packwithoutmask(const uint32_t *__restrict__ in,
                         __m128i *__restrict__ out, uint32_t bit);
void simd128unpack(const __m128i *__restrict__ in, uint32_t *__restrict__ out,
                uint32_t bit);

void SIMD128_fastunpack_32(const __m128i *__restrict__ in,
                        uint32_t *__restrict__ out, const uint32_t bit);
void SIMD128_fastpackwithoutmask_32(const uint32_t *__restrict__ in,
                                 __m128i *__restrict__ out, const uint32_t bit);
void SIMD128_fastpack_32(const uint32_t *__restrict__ in,
                      __m128i *__restrict__ out, const uint32_t bit);

} // namespace FastPFor

#endif /* SIMDBITPACKING128_H_ */
