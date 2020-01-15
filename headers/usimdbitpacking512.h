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

#ifndef USIMDBITPACKING512_H_
#define USIMDBITPACKING512_H_

#include "common.h"

namespace FastPForLib {

// We do not need to port this function to 256- and 512-bit SIMD
//void usimdpack(const uint32_t *__restrict__ in, __m512i *__restrict__ out,
//               uint32_t bit);
void usimd512packwithoutmask(const uint32_t *__restrict__ in,
                          __m512i *__restrict__ out, uint32_t bit);
void usimd512unpack(const __m512i *__restrict__ in, uint32_t *__restrict__ out,
                 uint32_t bit);

} // namespace FastPFor

#endif /* SIMDBITPACKING512_H_ */
