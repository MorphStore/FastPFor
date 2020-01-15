/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Daniel Lemire
 */

// ****************************************************************************
// This is a port of a modified version of the original SSE implementation to
// AVX-512. We do not claim that this code makes optimal use of AVX-512. Instead,
// it was deliberately ported in a straighforward way, mainly by substituting
// the SSE intrinsics for their AVX-512 equivalents. This required an increase
// of the block size. You can find the original implementation by Daniel Lemire
// et. al at https://github.com/lemire/FastPFOR .
// ****************************************************************************

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
