/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Daniel Lemire
 */
#ifdef __AVX2__

#include "simdbitpacking256.h"

namespace FastPForLib {

using namespace std;

static void SIMD256_nullunpacker32(const __m256i *__restrict__,
                                uint32_t *__restrict__ out) {
  // TODO put this constant in a central place
  const unsigned bitsPerByte = 8;
  memset(out, 0, sizeof(__m256i) * bitsPerByte * sizeof(uint32_t));
}

static void __SIMD256_fastpackwithoutmask1_32(const uint32_t *__restrict__ _in,
                                           __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask2_32(const uint32_t *__restrict__ _in,
                                           __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask3_32(const uint32_t *__restrict__ _in,
                                           __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 3 - 1);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 3 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask5_32(const uint32_t *__restrict__ _in,
                                           __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 5 - 3);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 5 - 1);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 5 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 5 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask6_32(const uint32_t *__restrict__ _in,
                                           __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 6 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 6 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 6 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 6 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask7_32(const uint32_t *__restrict__ _in,
                                           __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 7 - 3);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 7 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 7 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 7 - 5);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 7 - 1);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 7 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask9_32(const uint32_t *__restrict__ _in,
                                           __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 9 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 9 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 9 - 3);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 9 - 7);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 9 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 9 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 9 - 1);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 9 - 5);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask10_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 10 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 10 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 10 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 10 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 10 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 10 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 10 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 10 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask11_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 1);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 3);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 5);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 7);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 9);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask12_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 12 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 12 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 12 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 12 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 12 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 12 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 12 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 12 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask13_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 7);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 1);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 9);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 3);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 11);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 5);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask14_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask15_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 13);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 11);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 9);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 7);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 5);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 3);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 1);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 14);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask17_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 14);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 1);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 3);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 5);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 7);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 9);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 11);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 13);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 15);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask18_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 14);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 14);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask19_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 18);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 5);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 11);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 17);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 3);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 9);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 15);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 14);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 1);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 7);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 13);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask20_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask21_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 20);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 9);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 19);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 18);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 7);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 17);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 5);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 15);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 14);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 3);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 13);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 1);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 11);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask22_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 14);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 18);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 20);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 14);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 18);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 20);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask23_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 14);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 5);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 19);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 1);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 15);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 20);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 11);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 7);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 21);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 3);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 17);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 22);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 13);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 18);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 9);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask24_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask25_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 18);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 11);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 22);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 15);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 1);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 19);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 5);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 23);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 9);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 20);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 13);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 24);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 17);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 3);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 21);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 14);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 7);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask26_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 20);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 14);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 22);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 24);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 18);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 20);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 14);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 22);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 24);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 18);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask27_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 22);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 17);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 7);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 24);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 19);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 14);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 9);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 26);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 21);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 11);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 1);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 23);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 18);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 13);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 3);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 25);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 20);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 15);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 5);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask28_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 24);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 20);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 24);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 20);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 24);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 20);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 24);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 20);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask29_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 26);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 23);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 20);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 17);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 14);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 11);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 5);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 28);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 25);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 22);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 19);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 13);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 7);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 1);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 27);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 24);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 21);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 18);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 15);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 9);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 3);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask30_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 28);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 26);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 24);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 22);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 20);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 18);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 14);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 28);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 26);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 24);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 22);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 20);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 18);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 14);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask31_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 30);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 29);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 28);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 27);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 26);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 25);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 24);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 23);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 22);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 21);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 20);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 19);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 18);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 17);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 16);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 15);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 14);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 13);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 12);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 11);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 10);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 9);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 8);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 7);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 6);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 5);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 4);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 3);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 2);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 1);
  InReg = _mm256_load_si256(++in);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask32_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg = _mm256_load_si256(in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpackwithoutmask4_32(const uint32_t *__restrict__ _in,
                                           __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg;

  for (uint32_t outer = 0; outer < 4; ++outer) {
    InReg = _mm256_load_si256(in);
    OutReg = InReg;

    InReg = _mm256_load_si256(in + 1);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));

    InReg = _mm256_load_si256(in + 2);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));

    InReg = _mm256_load_si256(in + 3);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));

    InReg = _mm256_load_si256(in + 4);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));

    InReg = _mm256_load_si256(in + 5);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));

    InReg = _mm256_load_si256(in + 6);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));

    InReg = _mm256_load_si256(in + 7);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
    _mm256_store_si256(out, OutReg);
    ++out;

    in += 8;
  }
}

static void __SIMD256_fastpackwithoutmask8_32(const uint32_t *__restrict__ _in,
                                           __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg;

  for (uint32_t outer = 0; outer < 8; ++outer) {
    InReg = _mm256_load_si256(in);
    OutReg = InReg;

    InReg = _mm256_load_si256(in + 1);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));

    InReg = _mm256_load_si256(in + 2);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));

    InReg = _mm256_load_si256(in + 3);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
    _mm256_store_si256(out, OutReg);
    ++out;

    in += 4;
  }
}

static void __SIMD256_fastpackwithoutmask16_32(const uint32_t *__restrict__ _in,
                                            __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;
  __m256i InReg;

  for (uint32_t outer = 0; outer < 16; ++outer) {
    InReg = _mm256_load_si256(in);
    OutReg = InReg;

    InReg = _mm256_load_si256(in + 1);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
    _mm256_store_si256(out, OutReg);
    ++out;

    in += 2;
  }
}

static void __SIMD256_fastpack1_32(const uint32_t *__restrict__ _in,
                                __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 1) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack2_32(const uint32_t *__restrict__ _in,
                                __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 2) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack3_32(const uint32_t *__restrict__ _in,
                                __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 3) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 3 - 1);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 3 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack5_32(const uint32_t *__restrict__ _in,
                                __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 5) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 5 - 3);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 5 - 1);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 5 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 5 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack6_32(const uint32_t *__restrict__ _in,
                                __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 6) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 6 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 6 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 6 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 6 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack7_32(const uint32_t *__restrict__ _in,
                                __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 7) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 7 - 3);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 7 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 7 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 7 - 5);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 7 - 1);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 7 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack9_32(const uint32_t *__restrict__ _in,
                                __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 9) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 9 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 9 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 9 - 3);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 9 - 7);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 9 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 9 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 9 - 1);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 9 - 5);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack10_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 10) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 10 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 10 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 10 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 10 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 10 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 10 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 10 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 10 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack11_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 11) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 1);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 3);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 5);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 7);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 9);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 11 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack12_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 12) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 12 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 12 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 12 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 12 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 12 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 12 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 12 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 12 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack13_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 13) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 7);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 1);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 9);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 3);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 11);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 5);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 13 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack14_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 14) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 14 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack15_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 15) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 13);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 11);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 9);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 7);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 5);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 3);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 1);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 14);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 15 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack17_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 17) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 14);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 1);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 3);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 5);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 7);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 9);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 11);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 13);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 17 - 15);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack18_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 18) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 14);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 18 - 14);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack19_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 19) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 18);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 5);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 11);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 17);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 3);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 9);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 15);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 14);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 1);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 7);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 19 - 13);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack20_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 20) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 20 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack21_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 21) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 20);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 9);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 19);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 18);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 7);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 17);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 5);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 15);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 14);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 3);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 13);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 1);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 21 - 11);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack22_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 22) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 14);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 18);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 20);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 14);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 18);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 20);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 22 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack23_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 23) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 14);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 5);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 19);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 1);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 15);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 20);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 11);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 7);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 21);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 3);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 17);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 22);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 13);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 18);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 23 - 9);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack24_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 24) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 24 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack25_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 25) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 18);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 11);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 22);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 15);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 1);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 19);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 5);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 23);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 9);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 20);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 13);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 24);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 17);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 3);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 21);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 14);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 25 - 7);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack26_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 26) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 20);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 14);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 22);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 24);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 18);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 20);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 14);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 22);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 24);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 18);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 26 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack27_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 27) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 22);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 17);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 7);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 24);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 19);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 14);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 9);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 26);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 21);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 11);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 1);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 23);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 18);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 13);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 3);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 25);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 20);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 15);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 27 - 5);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack28_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 28) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 24);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 20);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 24);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 20);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 24);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 20);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 24);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 20);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 28 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack29_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 29) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 26);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 23);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 20);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 17);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 14);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 11);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 5);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 28);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 25);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 22);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 19);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 13);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 7);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 1);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 27);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 24);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 21);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 18);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 15);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 9);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 29 - 3);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack30_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 30) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 28);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 26);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 24);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 22);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 20);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 18);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 14);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 28);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 26);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 24);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 22);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 20);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 18);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 14);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 30 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack31_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  const __m256i mask = _mm256_set1_epi32((1U << 31) - 1);

  __m256i InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
  OutReg = InReg;
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 31));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 30);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 30));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 29);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 29));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 28);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 27);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 27));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 26);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 26));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 25);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 25));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 24);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 23);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 23));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 22);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 22));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 21);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 21));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 20);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 19);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 19));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 18);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 18));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 17);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 17));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 16);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 15);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 15));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 14);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 14));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 13);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 13));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 12);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 11);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 11));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 10);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 10));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 9);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 9));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 8);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 7);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 7));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 6);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 6));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 5);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 5));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 4);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 3);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 3));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 2);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 2));
  _mm256_store_si256(out, OutReg);
  ++out;
  OutReg = _mm256_srli_epi32(InReg, 31 - 1);
  InReg = _mm256_and_si256(_mm256_load_si256(++in), mask);

  OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 1));
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack32_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg;

  __m256i InReg = _mm256_load_si256(in);
  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
  ++out;
  InReg = _mm256_load_si256(++in);

  OutReg = InReg;
  _mm256_store_si256(out, OutReg);
}

static void __SIMD256_fastpack4_32(const uint32_t *__restrict__ _in,
                                __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg, InReg;
  const __m256i mask = _mm256_set1_epi32((1U << 4) - 1);

  for (uint32_t outer = 0; outer < 4; ++outer) {
    InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
    OutReg = InReg;

    InReg = _mm256_and_si256(_mm256_load_si256(in + 1), mask);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 4));

    InReg = _mm256_and_si256(_mm256_load_si256(in + 2), mask);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));

    InReg = _mm256_and_si256(_mm256_load_si256(in + 3), mask);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 12));

    InReg = _mm256_and_si256(_mm256_load_si256(in + 4), mask);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));

    InReg = _mm256_and_si256(_mm256_load_si256(in + 5), mask);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 20));

    InReg = _mm256_and_si256(_mm256_load_si256(in + 6), mask);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));

    InReg = _mm256_and_si256(_mm256_load_si256(in + 7), mask);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 28));
    _mm256_store_si256(out, OutReg);
    ++out;

    in += 8;
  }
}

static void __SIMD256_fastpack8_32(const uint32_t *__restrict__ _in,
                                __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg, InReg;
  const __m256i mask = _mm256_set1_epi32((1U << 8) - 1);

  for (uint32_t outer = 0; outer < 8; ++outer) {
    InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
    OutReg = InReg;

    InReg = _mm256_and_si256(_mm256_load_si256(in + 1), mask);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 8));

    InReg = _mm256_and_si256(_mm256_load_si256(in + 2), mask);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));

    InReg = _mm256_and_si256(_mm256_load_si256(in + 3), mask);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 24));
    _mm256_store_si256(out, OutReg);
    ++out;

    in += 4;
  }
}

static void __SIMD256_fastpack16_32(const uint32_t *__restrict__ _in,
                                 __m256i *__restrict__ out) {
  const __m256i *in = reinterpret_cast<const __m256i *>(_in);
  __m256i OutReg, InReg;
  const __m256i mask = _mm256_set1_epi32((1U << 16) - 1);

  for (uint32_t outer = 0; outer < 16; ++outer) {
    InReg = _mm256_and_si256(_mm256_load_si256(in), mask);
    OutReg = InReg;

    InReg = _mm256_and_si256(_mm256_load_si256(in + 1), mask);
    OutReg = _mm256_or_si256(OutReg, _mm256_slli_epi32(InReg, 16));
    _mm256_store_si256(out, OutReg);
    ++out;

    in += 2;
  }
}

static void __SIMD256_fastunpack1_32(const __m256i *__restrict__ in,
                                  uint32_t *__restrict__ _out) {
  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg1 = _mm256_load_si256(in);
  __m256i InReg2 = InReg1;
  __m256i OutReg1, OutReg2, OutReg3, OutReg4;
  const __m256i mask = _mm256_set1_epi32(1);

  unsigned shift = 0;

  for (unsigned i = 0; i < 8; ++i) {
    OutReg1 = _mm256_and_si256(_mm256_srli_epi32(InReg1, shift++), mask);
    OutReg2 = _mm256_and_si256(_mm256_srli_epi32(InReg2, shift++), mask);
    OutReg3 = _mm256_and_si256(_mm256_srli_epi32(InReg1, shift++), mask);
    OutReg4 = _mm256_and_si256(_mm256_srli_epi32(InReg2, shift++), mask);
    _mm256_store_si256(out++, OutReg1);
    _mm256_store_si256(out++, OutReg2);
    _mm256_store_si256(out++, OutReg3);
    _mm256_store_si256(out++, OutReg4);
  }
}

static void __SIMD256_fastunpack2_32(const __m256i *__restrict__ in,
                                  uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 2) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 14), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 18), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 20), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 22), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 24), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 26), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 28), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 14), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 18), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 20), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 22), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 24), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 26), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 28), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack3_32(const __m256i *__restrict__ in,
                                  uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 3) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 3), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 9), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 15), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 18), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 21), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 24), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 27), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 3 - 1), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 1), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 7), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 13), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 19), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 22), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 25), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 28), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 31);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 3 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 5), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 11), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 14), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 17), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 20), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 23), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 26), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 29);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack4_32(const __m256i *__restrict__ in,
                                  uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 4) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 20), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 24), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 20), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 24), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 20), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 24), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 20), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 24), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack5_32(const __m256i *__restrict__ in,
                                  uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 5) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 5), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 15), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 20), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 25), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 5 - 3), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 3), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 13), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 18), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 23), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 5 - 1), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 1), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 11), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 21), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 26), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 31);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 5 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 9), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 14), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 19), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 24), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 29);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 5 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 7), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 17), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 22), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 27);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack6_32(const __m256i *__restrict__ in,
                                  uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 6) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 18), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 24), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 6 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 22), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 6 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 14), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 20), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 18), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 24), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 6 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 22), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 6 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 14), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 20), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack7_32(const __m256i *__restrict__ in,
                                  uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 7) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 7), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 14), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 21), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 7 - 3), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 3), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 17), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 24), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 31);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 7 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 13), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 20), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 27);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 7 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 9), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 23), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 7 - 5), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 5), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 19), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 7 - 1), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 1), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 15), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 22), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 29);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 7 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 11), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 18), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 25);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack8_32(const __m256i *__restrict__ in,
                                  uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 8) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack9_32(const __m256i *__restrict__ in,
                                  uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 9) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 9), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 18), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 27);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 9 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 13), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 22), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 31);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 9 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 17), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 9 - 3), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 3), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 21), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 9 - 7), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 7), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 25);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 9 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 11), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 20), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 29);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 9 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 15), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 9 - 1), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 1), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 19), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 9 - 5), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 5), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 14), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 23);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack10_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 10) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 20), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 10 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 18), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 10 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 10 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 14), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 10 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 20), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 10 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 18), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 10 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 10 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 14), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 10 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack11_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 11) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 11), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 11 - 1), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 1), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 23);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 11 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 13), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 11 - 3), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 3), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 14), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 25);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 11 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 15), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 11 - 5), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 5), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 27);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 11 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 17), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 11 - 7), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 7), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 18), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 29);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 11 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 19), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 11 - 9), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 9), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 20), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 31);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 11 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 21);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack12_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 12) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 12 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 12 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 12 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 12 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 12 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 12 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 12 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 12 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack13_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 13) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 13), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 13 - 7), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 7), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 13 - 1), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 1), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 14), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 27);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 13 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 21);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 13 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 15), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 13 - 9), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 9), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 13 - 3), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 3), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 29);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 13 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 23);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 13 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 17), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 13 - 11), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 11), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 13 - 5), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 5), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 18), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 31);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 13 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 25);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 13 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 19);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack14_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 14) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 14), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 14 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 14 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 14 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 14 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 14 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 14 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 14), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 14 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 14 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 14 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 14 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 14 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 14 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack15_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 15) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 15), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 15 - 13), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 13), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 15 - 11), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 11), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 15 - 9), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 9), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 15 - 7), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 7), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 15 - 5), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 5), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 15 - 3), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 3), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 15 - 1), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 1), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 16), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 31);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 15 - 14), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 14), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 29);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 15 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 27);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 15 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 25);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 15 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 23);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 15 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 21);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 15 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 19);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 15 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 17);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack16_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 16) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack17_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 17) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 17);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 17 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 19);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 17 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 21);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 17 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 23);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 17 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 25);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 17 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 27);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 17 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 29);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 17 - 14), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 14), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 31);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 17 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 17 - 1), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 1), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 17 - 3), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 3), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 17 - 5), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 5), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 17 - 7), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 7), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 17 - 9), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 9), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 17 - 11), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 11), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 17 - 13), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 13), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 17 - 15), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 15);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack18_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 18) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 18 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 18 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 18 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 18 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 18 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 18 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 18 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 18 - 14), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 14);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 18 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 18 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 18 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 18 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 18 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 18 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 18 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 18 - 14), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 14);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack19_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 19) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 19);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 19 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 25);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 19 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 12), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 31);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 19 - 18), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 19 - 5), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 5), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 19 - 11), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 11), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 19 - 17), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 17);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 19 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 23);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 19 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 29);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 19 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 19 - 3), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 3), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 19 - 9), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 9), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 19 - 15), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 15);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 19 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 21);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 19 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 27);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 19 - 14), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 14);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 19 - 1), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 1), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 19 - 7), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 7), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 19 - 13), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 13);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack20_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 20) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 20 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 20 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 20 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 20 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 20 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 20 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 20 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 20 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 20 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 20 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 20 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 20 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 20 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 20 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 20 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 20 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack21_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 21) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 21);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 10), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 31);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 20), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 9), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 9), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 19), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 19);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 29);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 18), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 7), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 7), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 17), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 17);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 27);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 5), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 5), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 15), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 15);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 25);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 14), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 14);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 3), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 3), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 13), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 13);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 23);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 1), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 1), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 21 - 11), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 11);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack22_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 22) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 14), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 14);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 18), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 20), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 10);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 14), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 14);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 18), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 20), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 22 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 10);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack23_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 23) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 23);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 14), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 14);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 5), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 5), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 19), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 19);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 10);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 1), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 1), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 15), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 15);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 29);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 20), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 11), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 11);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 25);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 7), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 7), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 21), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 21);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 3), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 3), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 17), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 17);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 8), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 31);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 22), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 13), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 13);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 27);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 18), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 23 - 9), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 9);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack24_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 24) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 24 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 24 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 24 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 24 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 24 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 24 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 24 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 24 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 24 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 24 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 24 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 24 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 24 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 24 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 24 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 24 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack25_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 25) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 25);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 18), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 11), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 11);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 29);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 22), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 15), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 15);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 1), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 1), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 19), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 19);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 5), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 5), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 23), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 23);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 9), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 9);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 27);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 20), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 13), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 13);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 6), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 31);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 24), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 17), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 17);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 10);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 3), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 3), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 21), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 21);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 14), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 14);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 25 - 7), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 7);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack26_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 26) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 20), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 14), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 14);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 22), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 10);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 24), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 18), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 6);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 20), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 14), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 14);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 22), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 10);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 24), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 18), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 26 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 6);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack27_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 27) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 27);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 22), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 17), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 17);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 7), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 7);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 29);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 24), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 19), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 19);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 14), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 14);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 9), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 9);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 4), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 31);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 26), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 21), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 21);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 11), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 11);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 6);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 1), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 1), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 23), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 23);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 18), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 13), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 13);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 3), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 3), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 25), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 25);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 20), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 15), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 15);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 10);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 27 - 5), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 5);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack28_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 28) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 24), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 20), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 4);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 24), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 20), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 4);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 24), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 20), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 4);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 24), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 20), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 28 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 4);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack29_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 29) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 29);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 26), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 23), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 23);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 20), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 17), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 17);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 14), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 14);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 11), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 11);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 5), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 5);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 2), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 31);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 28), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 25), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 25);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 22), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 19), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 19);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 13), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 13);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 10);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 7), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 7);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 4);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 1), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(_mm256_srli_epi32(InReg, 1), mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 27), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 27);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 24), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 21), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 21);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 18), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 15), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 15);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 9), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 9);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 6);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 29 - 3), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 3);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack30_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 30) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 28), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 26), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 24), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 22), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 20), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 18), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 14), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 14);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 10);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 6);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 4);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 2);
  InReg = _mm256_load_si256(++in);

  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 28), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 26), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 24), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 22), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 20), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 18), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 14), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 14);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 10);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 6);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 4);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 30 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 2);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack31_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {

  __m256i *out = reinterpret_cast<__m256i *>(_out);
  __m256i InReg = _mm256_load_si256(in);
  __m256i OutReg;
  const __m256i mask = _mm256_set1_epi32((1U << 31) - 1);

  OutReg = _mm256_and_si256(InReg, mask);
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 31);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 30), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 30);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 29), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 29);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 28), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 28);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 27), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 27);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 26), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 26);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 25), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 25);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 24), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 24);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 23), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 23);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 22), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 22);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 21), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 21);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 20), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 20);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 19), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 19);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 18), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 18);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 17), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 17);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 16), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 16);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 15), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 15);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 14), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 14);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 13), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 13);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 12), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 12);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 11), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 11);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 10), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 10);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 9), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 9);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 8), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 8);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 7), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 7);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 6), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 6);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 5), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 5);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 4), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 4);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 3), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 3);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 2), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 2);
  InReg = _mm256_load_si256(++in);

  OutReg =
      _mm256_or_si256(OutReg, _mm256_and_si256(_mm256_slli_epi32(InReg, 31 - 1), mask));
  _mm256_store_si256(out++, OutReg);

  OutReg = _mm256_srli_epi32(InReg, 1);
  _mm256_store_si256(out++, OutReg);
}

static void __SIMD256_fastunpack32_32(const __m256i *__restrict__ in,
                                   uint32_t *__restrict__ _out) {
  __m256i *out = reinterpret_cast<__m256i *>(_out);
  for (uint32_t outer = 0; outer < 32; ++outer) {
    _mm256_store_si256(out++, _mm256_load_si256(in++));
  }
}

void simd256unpack(const __m256i *__restrict__ in, uint32_t *__restrict__ out,
                const uint32_t bit) {
  switch (bit) {
  case 0:
    SIMD256_nullunpacker32(in, out);
    return;

  case 1:
    __SIMD256_fastunpack1_32(in, out);
    return;

  case 2:
    __SIMD256_fastunpack2_32(in, out);
    return;

  case 3:
    __SIMD256_fastunpack3_32(in, out);
    return;

  case 4:
    __SIMD256_fastunpack4_32(in, out);
    return;

  case 5:
    __SIMD256_fastunpack5_32(in, out);
    return;

  case 6:
    __SIMD256_fastunpack6_32(in, out);
    return;

  case 7:
    __SIMD256_fastunpack7_32(in, out);
    return;

  case 8:
    __SIMD256_fastunpack8_32(in, out);
    return;

  case 9:
    __SIMD256_fastunpack9_32(in, out);
    return;

  case 10:
    __SIMD256_fastunpack10_32(in, out);
    return;

  case 11:
    __SIMD256_fastunpack11_32(in, out);
    return;

  case 12:
    __SIMD256_fastunpack12_32(in, out);
    return;

  case 13:
    __SIMD256_fastunpack13_32(in, out);
    return;

  case 14:
    __SIMD256_fastunpack14_32(in, out);
    return;

  case 15:
    __SIMD256_fastunpack15_32(in, out);
    return;

  case 16:
    __SIMD256_fastunpack16_32(in, out);
    return;

  case 17:
    __SIMD256_fastunpack17_32(in, out);
    return;

  case 18:
    __SIMD256_fastunpack18_32(in, out);
    return;

  case 19:
    __SIMD256_fastunpack19_32(in, out);
    return;

  case 20:
    __SIMD256_fastunpack20_32(in, out);
    return;

  case 21:
    __SIMD256_fastunpack21_32(in, out);
    return;

  case 22:
    __SIMD256_fastunpack22_32(in, out);
    return;

  case 23:
    __SIMD256_fastunpack23_32(in, out);
    return;

  case 24:
    __SIMD256_fastunpack24_32(in, out);
    return;

  case 25:
    __SIMD256_fastunpack25_32(in, out);
    return;

  case 26:
    __SIMD256_fastunpack26_32(in, out);
    return;

  case 27:
    __SIMD256_fastunpack27_32(in, out);
    return;

  case 28:
    __SIMD256_fastunpack28_32(in, out);
    return;

  case 29:
    __SIMD256_fastunpack29_32(in, out);
    return;

  case 30:
    __SIMD256_fastunpack30_32(in, out);
    return;

  case 31:
    __SIMD256_fastunpack31_32(in, out);
    return;

  case 32:
    __SIMD256_fastunpack32_32(in, out);
    return;

  default:
    break;
  }
  throw logic_error("number of bits is unsupported");
}

/*assumes that integers fit in the prescribed number of bits*/
void simd256packwithoutmask(const uint32_t *__restrict__ in,
                         __m256i *__restrict__ out, const uint32_t bit) {
  switch (bit) {
  case 0:
    return;

  case 1:
    __SIMD256_fastpackwithoutmask1_32(in, out);
    return;

  case 2:
    __SIMD256_fastpackwithoutmask2_32(in, out);
    return;

  case 3:
    __SIMD256_fastpackwithoutmask3_32(in, out);
    return;

  case 4:
    __SIMD256_fastpackwithoutmask4_32(in, out);
    return;

  case 5:
    __SIMD256_fastpackwithoutmask5_32(in, out);
    return;

  case 6:
    __SIMD256_fastpackwithoutmask6_32(in, out);
    return;

  case 7:
    __SIMD256_fastpackwithoutmask7_32(in, out);
    return;

  case 8:
    __SIMD256_fastpackwithoutmask8_32(in, out);
    return;

  case 9:
    __SIMD256_fastpackwithoutmask9_32(in, out);
    return;

  case 10:
    __SIMD256_fastpackwithoutmask10_32(in, out);
    return;

  case 11:
    __SIMD256_fastpackwithoutmask11_32(in, out);
    return;

  case 12:
    __SIMD256_fastpackwithoutmask12_32(in, out);
    return;

  case 13:
    __SIMD256_fastpackwithoutmask13_32(in, out);
    return;

  case 14:
    __SIMD256_fastpackwithoutmask14_32(in, out);
    return;

  case 15:
    __SIMD256_fastpackwithoutmask15_32(in, out);
    return;

  case 16:
    __SIMD256_fastpackwithoutmask16_32(in, out);
    return;

  case 17:
    __SIMD256_fastpackwithoutmask17_32(in, out);
    return;

  case 18:
    __SIMD256_fastpackwithoutmask18_32(in, out);
    return;

  case 19:
    __SIMD256_fastpackwithoutmask19_32(in, out);
    return;

  case 20:
    __SIMD256_fastpackwithoutmask20_32(in, out);
    return;

  case 21:
    __SIMD256_fastpackwithoutmask21_32(in, out);
    return;

  case 22:
    __SIMD256_fastpackwithoutmask22_32(in, out);
    return;

  case 23:
    __SIMD256_fastpackwithoutmask23_32(in, out);
    return;

  case 24:
    __SIMD256_fastpackwithoutmask24_32(in, out);
    return;

  case 25:
    __SIMD256_fastpackwithoutmask25_32(in, out);
    return;

  case 26:
    __SIMD256_fastpackwithoutmask26_32(in, out);
    return;

  case 27:
    __SIMD256_fastpackwithoutmask27_32(in, out);
    return;

  case 28:
    __SIMD256_fastpackwithoutmask28_32(in, out);
    return;

  case 29:
    __SIMD256_fastpackwithoutmask29_32(in, out);
    return;

  case 30:
    __SIMD256_fastpackwithoutmask30_32(in, out);
    return;

  case 31:
    __SIMD256_fastpackwithoutmask31_32(in, out);
    return;

  case 32:
    __SIMD256_fastpackwithoutmask32_32(in, out);
    return;

  default:
    break;
  }
  throw logic_error("number of bits is unsupported");
}

/*assumes that integers fit in the prescribed number of bits*/
void simd256pack(const uint32_t *__restrict__ in, __m256i *__restrict__ out,
              const uint32_t bit) {
  switch (bit) {
  case 0:
    return;

  case 1:
    __SIMD256_fastpack1_32(in, out);
    return;

  case 2:
    __SIMD256_fastpack2_32(in, out);
    return;

  case 3:
    __SIMD256_fastpack3_32(in, out);
    return;

  case 4:
    __SIMD256_fastpack4_32(in, out);
    return;

  case 5:
    __SIMD256_fastpack5_32(in, out);
    return;

  case 6:
    __SIMD256_fastpack6_32(in, out);
    return;

  case 7:
    __SIMD256_fastpack7_32(in, out);
    return;

  case 8:
    __SIMD256_fastpack8_32(in, out);
    return;

  case 9:
    __SIMD256_fastpack9_32(in, out);
    return;

  case 10:
    __SIMD256_fastpack10_32(in, out);
    return;

  case 11:
    __SIMD256_fastpack11_32(in, out);
    return;

  case 12:
    __SIMD256_fastpack12_32(in, out);
    return;

  case 13:
    __SIMD256_fastpack13_32(in, out);
    return;

  case 14:
    __SIMD256_fastpack14_32(in, out);
    return;

  case 15:
    __SIMD256_fastpack15_32(in, out);
    return;

  case 16:
    __SIMD256_fastpack16_32(in, out);
    return;

  case 17:
    __SIMD256_fastpack17_32(in, out);
    return;

  case 18:
    __SIMD256_fastpack18_32(in, out);
    return;

  case 19:
    __SIMD256_fastpack19_32(in, out);
    return;

  case 20:
    __SIMD256_fastpack20_32(in, out);
    return;

  case 21:
    __SIMD256_fastpack21_32(in, out);
    return;

  case 22:
    __SIMD256_fastpack22_32(in, out);
    return;

  case 23:
    __SIMD256_fastpack23_32(in, out);
    return;

  case 24:
    __SIMD256_fastpack24_32(in, out);
    return;

  case 25:
    __SIMD256_fastpack25_32(in, out);
    return;

  case 26:
    __SIMD256_fastpack26_32(in, out);
    return;

  case 27:
    __SIMD256_fastpack27_32(in, out);
    return;

  case 28:
    __SIMD256_fastpack28_32(in, out);
    return;

  case 29:
    __SIMD256_fastpack29_32(in, out);
    return;

  case 30:
    __SIMD256_fastpack30_32(in, out);
    return;

  case 31:
    __SIMD256_fastpack31_32(in, out);
    return;

  case 32:
    __SIMD256_fastpack32_32(in, out);
    return;

  default:
    break;
  }
  throw logic_error("number of bits is unsupported");
}

void SIMD256_fastunpack_32(const __m256i *__restrict__ in,
                        uint32_t *__restrict__ out, const uint32_t bit) {
  simd256unpack(in, out, bit);
}
void SIMD256_fastpackwithoutmask_32(const uint32_t *__restrict__ in,
                                 __m256i *__restrict__ out,
                                 const uint32_t bit) {
  simd256packwithoutmask(in, out, bit);
}
void SIMD256_fastpack_32(const uint32_t *__restrict__ in,
                      __m256i *__restrict__ out, const uint32_t bit) {
  simd256pack(in, out, bit);
}

} // namespace FastPFor

#endif // __AVX2__