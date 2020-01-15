/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Daniel Lemire, http://lemire.me/en/
 */

#ifndef SIMDBINARYPACKING512_H_
#define SIMDBINARYPACKING512_H_

#include "codecs.h"
#include "simdbitpacking512.h"
#include "util.h"

namespace FastPForLib {

/**
 *
 * Designed by D. Lemire with ideas from Leonid Boystov. This scheme is NOT
 * patented.
 *
 * Compresses data in blocks of 512 integers.
 * Uses 512-bit AVX-512 SIMD instructions.
 *
 * Reference and documentation:
 *
 * Daniel Lemire and Leonid Boytsov, Decoding billions of integers per second
 * through vectorization
 * http://arxiv.org/abs/1209.2137
 */
class SIMDBinaryPacking512 : public IntegerCODEC {
public:
  static const uint32_t CookiePadder = 123456;
  // TODO put this constant in a central place
  static const unsigned bitsPerByte = 8;
  static const uint32_t MiniBlockSize = sizeof(__m512i) * bitsPerByte;
  static const uint32_t HowManyMiniBlocks = sizeof(__m512i);
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
    while (needPaddingTo64bytes(out))
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
      *out++ = (Bs[32] << 24) | (Bs[33] << 16) | (Bs[34] << 8) | Bs[35];
      *out++ = (Bs[36] << 24) | (Bs[37] << 16) | (Bs[38] << 8) | Bs[39];
      *out++ = (Bs[40] << 24) | (Bs[41] << 16) | (Bs[42] << 8) | Bs[43];
      *out++ = (Bs[44] << 24) | (Bs[45] << 16) | (Bs[46] << 8) | Bs[47];
      *out++ = (Bs[48] << 24) | (Bs[49] << 16) | (Bs[50] << 8) | Bs[51];
      *out++ = (Bs[52] << 24) | (Bs[53] << 16) | (Bs[54] << 8) | Bs[55];
      *out++ = (Bs[56] << 24) | (Bs[57] << 16) | (Bs[58] << 8) | Bs[59];
      *out++ = (Bs[60] << 24) | (Bs[61] << 16) | (Bs[62] << 8) | Bs[63];
      for (uint32_t i = 0; i < HowManyMiniBlocks; ++i) {
        SIMD512_fastpackwithoutmask_32(in + i * MiniBlockSize,
                                    reinterpret_cast<__m512i *>(out), Bs[i]);
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
      *out++ = (Bs[32] << 24) | (Bs[33] << 16) | (Bs[34] << 8) | Bs[35];
      *out++ = (Bs[36] << 24) | (Bs[37] << 16) | (Bs[38] << 8) | Bs[39];
      *out++ = (Bs[40] << 24) | (Bs[41] << 16) | (Bs[42] << 8) | Bs[43];
      *out++ = (Bs[44] << 24) | (Bs[45] << 16) | (Bs[46] << 8) | Bs[47];
      *out++ = (Bs[48] << 24) | (Bs[49] << 16) | (Bs[50] << 8) | Bs[51];
      *out++ = (Bs[52] << 24) | (Bs[53] << 16) | (Bs[54] << 8) | Bs[55];
      *out++ = (Bs[56] << 24) | (Bs[57] << 16) | (Bs[58] << 8) | Bs[59];
      *out++ = (Bs[60] << 24) | (Bs[61] << 16) | (Bs[62] << 8) | Bs[63];
      for (uint32_t i = 0; i < howmany; ++i) {
        SIMD512_fastpackwithoutmask_32(in + i * MiniBlockSize,
                                    reinterpret_cast<__m512i *>(out), Bs[i]);
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
    if (needPaddingTo64bytes(out))
      throw std::runtime_error("bad initial output align");
    while (needPaddingTo64bytes(in)) {
      if (in[0] != CookiePadder)
        throw std::logic_error("SIMDBinaryPacking512 alignment issue.");
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
        SIMD512_fastunpack_32(reinterpret_cast<const __m512i *>(in),
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
        SIMD512_fastunpack_32(reinterpret_cast<const __m512i *>(in),
                           out + i * MiniBlockSize, Bs[i]);
        in += MiniBlockSize / 32 * Bs[i];
      }
      out += howmany * MiniBlockSize;
      assert(out == initout + actuallength);
    }
    nvalue = out - initout;
    return in;
  }

  std::string name() const { return "SIMDBinaryPacking512"; }
};

} // namespace FastPFor

#endif /* SIMDBINARYPACKING512_H_ */
