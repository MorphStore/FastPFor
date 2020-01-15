/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Daniel Lemire
 */
#ifndef USIMDBITPACKING256_H_
#define USIMDBITPACKING256_H_

#include "common.h"

namespace FastPForLib {

// We do not need to port this function to 256- and 512-bit SIMD
//void usimdpack(const uint32_t *__restrict__ in, __m256i *__restrict__ out,
//               uint32_t bit);
void usimd256packwithoutmask(const uint32_t *__restrict__ in,
                          __m256i *__restrict__ out, uint32_t bit);
void usimd256unpack(const __m256i *__restrict__ in, uint32_t *__restrict__ out,
                 uint32_t bit);

} // namespace FastPFor

#endif /* SIMDBITPACKING256_H_ */
