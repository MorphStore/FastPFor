/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Daniel Lemire, http://lemire.me/en/
 */

// ****************************************************************************
// This is a port of a modified version of the original SSE implementation to
// AVX2. We do not claim that this code makes optimal use of AVX2. Instead,
// it was deliberately ported in a straighforward way, mainly by substituting
// the SSE intrinsics for their AVX2 equivalents. This required an increase
// of the block size. You can find the original implementation by Daniel Lemire
// et. al at https://github.com/lemire/FastPFOR .
// ****************************************************************************

#ifndef SIMDBINARYPACKING256_H_
#define SIMDBINARYPACKING256_H_

#include "codecs.h"
#include "simdbitpacking256.h"
#include "util.h"

namespace FastPForLib {

/**
 *
 * Designed by D. Lemire with ideas from Leonid Boystov. This scheme is NOT
 * patented.
 *
 * Compresses data in blocks of 256 integers.
 * Uses 256-bit AVX2 SIMD instructions.
 *
 * Reference and documentation:
 *
 * Daniel Lemire and Leonid Boytsov, Decoding billions of integers per second
 * through vectorization
 * http://arxiv.org/abs/1209.2137
 */
class SIMDBinaryPacking256 : public IntegerCODEC {
public:
  static const uint32_t CookiePadder = 123456;
  // TODO put this constant in a central place
  static const unsigned bitsPerByte = 8;
  static const uint32_t MiniBlockSize = sizeof(__m256i) * bitsPerByte;
  static const uint32_t HowManyMiniBlocks = sizeof(__m256i);
  static const uint32_t BlockSize = MiniBlockSize;

  /**
   * The way this code is written, it will automatically "pad" the
   * header according to the alignment of the out pointer. So if you
   * move the data around, you should preserve the alignment.
   */
  void encodeArray(const uint32_t *in, const size_t length, uint32_t *out,
                   size_t &nvalue) {
    checkifdivisibleby(length, BlockSize);
    const uint32_t *const initout(out);
    *out++ = static_cast<uint32_t>(length);
    // TODO why doesn't this work? I manually inlined the function.
#if 0
    while (needPaddingTo256Bits(out))
#else
    while ((reinterpret_cast<uintptr_t>(out) & 31) != 0)
#endif
      *out++ = CookiePadder;
    uint32_t Bs[HowManyMiniBlocks];
    const uint32_t *const final = in + length;
    for (; in + HowManyMiniBlocks * MiniBlockSize <= final;
         in += HowManyMiniBlocks * MiniBlockSize) {

      for (uint32_t i = 0; i < HowManyMiniBlocks; ++i)
        Bs[i] = maxbits(in + i * MiniBlockSize, in + (i + 1) * MiniBlockSize);
      *out++ = (Bs[ 0] << 24) | (Bs[ 1] << 16) | (Bs[ 2] << 8) | Bs[ 3];
      *out++ = (Bs[ 4] << 24) | (Bs[ 5] << 16) | (Bs[ 6] << 8) | Bs[ 7];
      *out++ = (Bs[ 8] << 24) | (Bs[ 9] << 16) | (Bs[10] << 8) | Bs[11];
      *out++ = (Bs[12] << 24) | (Bs[13] << 16) | (Bs[14] << 8) | Bs[15];
      *out++ = (Bs[16] << 24) | (Bs[17] << 16) | (Bs[18] << 8) | Bs[19];
      *out++ = (Bs[20] << 24) | (Bs[21] << 16) | (Bs[22] << 8) | Bs[23];
      *out++ = (Bs[24] << 24) | (Bs[25] << 16) | (Bs[26] << 8) | Bs[27];
      *out++ = (Bs[28] << 24) | (Bs[29] << 16) | (Bs[30] << 8) | Bs[31];
      for (uint32_t i = 0; i < HowManyMiniBlocks; ++i) {
        SIMD256_fastpackwithoutmask_32(in + i * MiniBlockSize,
                                    reinterpret_cast<__m256i *>(out), Bs[i]);
        out += MiniBlockSize / 32 * Bs[i];
      }
    }
    if (in < final) {
      const size_t howmany = (final - in) / MiniBlockSize;
      memset(&Bs[0], 0, HowManyMiniBlocks * sizeof(uint32_t));
      for (uint32_t i = 0; i < howmany; ++i)
        Bs[i] = maxbits(in + i * MiniBlockSize, in + (i + 1) * MiniBlockSize);
      *out++ = (Bs[ 0] << 24) | (Bs[ 1] << 16) | (Bs[ 2] << 8) | Bs[ 3];
      *out++ = (Bs[ 4] << 24) | (Bs[ 5] << 16) | (Bs[ 6] << 8) | Bs[ 7];
      *out++ = (Bs[ 8] << 24) | (Bs[ 9] << 16) | (Bs[10] << 8) | Bs[11];
      *out++ = (Bs[12] << 24) | (Bs[13] << 16) | (Bs[14] << 8) | Bs[15];
      *out++ = (Bs[16] << 24) | (Bs[17] << 16) | (Bs[18] << 8) | Bs[19];
      *out++ = (Bs[20] << 24) | (Bs[21] << 16) | (Bs[22] << 8) | Bs[23];
      *out++ = (Bs[24] << 24) | (Bs[25] << 16) | (Bs[26] << 8) | Bs[27];
      *out++ = (Bs[28] << 24) | (Bs[29] << 16) | (Bs[30] << 8) | Bs[31];
      for (uint32_t i = 0; i < howmany; ++i) {
        SIMD256_fastpackwithoutmask_32(in + i * MiniBlockSize,
                                    reinterpret_cast<__m256i *>(out), Bs[i]);
        out += MiniBlockSize / 32 * Bs[i];
      }
      in += howmany * MiniBlockSize;
      assert(in == final);
    }

    nvalue = out - initout;
  }

  const uint32_t *decodeArray(const uint32_t *in, const size_t /*length*/,
                              uint32_t *out, size_t &nvalue) {
    const uint32_t actuallength = *in++;
    // TODO why doesn't this work? I manually inlined the function.
#if 0
    if (needPaddingTo256Bits(out))
#else
    if ((reinterpret_cast<uintptr_t>(out) & 31) != 0)
#endif
      throw std::runtime_error("bad initial output align");
    // TODO why doesn't this work? I manually inlined the function.
#if 0
    while (needPaddingTo256Bits(in)) {
#else
    while ((reinterpret_cast<uintptr_t>(in) & 31) != 0) {
#endif
      if (in[0] != CookiePadder)
        throw std::logic_error("SIMDBinaryPacking256 alignment issue.");
      ++in;
    }
    const uint32_t *const initout(out);
    uint32_t Bs[HowManyMiniBlocks];
    for (; out < initout +
                     actuallength / (HowManyMiniBlocks * MiniBlockSize) *
                         HowManyMiniBlocks * MiniBlockSize;
         out += HowManyMiniBlocks * MiniBlockSize) {
      for (uint32_t i = 0; i < HowManyMiniBlocks / sizeof(uint32_t); ++i, ++in) {
        Bs[0 + sizeof(uint32_t) * i] = static_cast<uint8_t>(in[0] >> 24);
        Bs[1 + sizeof(uint32_t) * i] = static_cast<uint8_t>(in[0] >> 16);
        Bs[2 + sizeof(uint32_t) * i] = static_cast<uint8_t>(in[0] >> 8);
        Bs[3 + sizeof(uint32_t) * i] = static_cast<uint8_t>(in[0]);
      }
      for (uint32_t i = 0; i < HowManyMiniBlocks; ++i) {
        // D.L. : is the reinterpret_cast safe here?
        SIMD256_fastunpack_32(reinterpret_cast<const __m256i *>(in),
                           out + i * MiniBlockSize, Bs[i]);
        in += MiniBlockSize / 32 * Bs[i];
      }
    }
    if (out < initout + actuallength) {
      const size_t howmany = (initout + actuallength - out) / MiniBlockSize;
      for (uint32_t i = 0; i < HowManyMiniBlocks / sizeof(uint32_t); ++i, ++in) {
        Bs[0 + sizeof(uint32_t) * i] = static_cast<uint8_t>(in[0] >> 24);
        Bs[1 + sizeof(uint32_t) * i] = static_cast<uint8_t>(in[0] >> 16);
        Bs[2 + sizeof(uint32_t) * i] = static_cast<uint8_t>(in[0] >> 8);
        Bs[3 + sizeof(uint32_t) * i] = static_cast<uint8_t>(in[0]);
      }
      for (uint32_t i = 0; i < howmany; ++i) {
        SIMD256_fastunpack_32(reinterpret_cast<const __m256i *>(in),
                           out + i * MiniBlockSize, Bs[i]);
        in += MiniBlockSize / 32 * Bs[i];
      }
      out += howmany * MiniBlockSize;
      assert(out == initout + actuallength);
    }
    nvalue = out - initout;
    return in;
  }

  std::string name() const { return "SIMDBinaryPacking256"; }
};

} // namespace FastPFor

#endif /* SIMDBINARYPACKING256_H_ */
