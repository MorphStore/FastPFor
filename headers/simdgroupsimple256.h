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

#ifndef SIMDGROUPSIMPLE256_H_
#define SIMDGROUPSIMPLE256_H_

#include "common.h"
#include "codecs.h"
#include "util.h"

namespace FastPForLib {

/**
 * This is an implementation of the compression algorithm SIMD-GroupSimple,
 * which was proposed in Section 4 of the following paper:
 * 
 * W. X. Zhao, X. Zhang, D. Lemire, D. Shan, J. Nie, H. Yan, and J. Wen.
 * A general simd-based approach to accelerating compression algorithms.
 * ACM Trans. Inf. Syst., 33(3), 2015.
 * http://arxiv.org/abs/1502.01916
 * 
 * Implemented by Patrick Damme,
 * https://wwwdb.inf.tu-dresden.de/our-group/team/patrick-damme . 
 * 
 * We provide two variants of the compression part of the algorithm.
 * 
 * The original variant
 * ====================
 * The first variant closely follows the original algorithm as described in the
 * paper. We also implemented the two optimizations mentioned in the paper:
 * - The calculation of pseudo group max values instead of group max values.
 * - One specialized (un)packing routine for each selector, whereby the
 *   appropriate routine is selected by a switch-case-statement.
 * However, our implementation differs from the paper in some minor points, for
 * instance, we directly look up the mask used in the pattern selection
 * algorithm instead of calculating it from a looked up bit width.
 * 
 * The variant using a group max ring buffer
 * =========================================
 * The second variant is based on the original description, but uses a ring
 * buffer instead of an array for the (pseudo) group max values to reduce the
 * size of the temporary data during the compression. More details on this can
 * be found in Section 3.2.3 of the following paper:
 * 
 * P. Damme, D. Habich, J. Hildebrandt, and W. Lehner. Lightweight data
 * compression algorithms: An experimental survey (experiments and analyses).
 * In Proceedings of the 20th International Conference on Extending Database
 * Technology, EDBT 2017.
 * http://openproceedings.org/2017/conf/edbt/paper-146.pdf
 * 
 * The template parameter useRingBuf determines which variant is used:
 * - false: original variant
 * - true: the variant with the ring buffer
 * Both variants use the same packing routines and the same decompression
 * algorithm. Our experiments suggest that the variant with the ring buffer is
 * faster than the original algorithm for small bit widths.
 * 
 * Compressed data format
 * ======================
 * As described in the original paper, the compressed data consists of two
 * areas, whose separation is a crucial point of the algorithm: the selectors
 * area and the data area, which we store in this order. The original variant
 * generates all selectors before it compresses the blocks. Thus, it knows the
 * size of the selectors area before it starts writing to the data area. So it
 * can start the data area directly after the selectors area. However, the
 * variant using the ring buffer compresses a block immediately when it
 * determines the selector. Thus, it does not know the size of the selectors
 * area before it starts writing to the data area. To prevent the two areas
 * from overlapping, we need to leave a "pessimistic gap" between them, i.e.,
 * we reserve the worst-case number of bytes for the selectors area. This has
 * no impact on the amount of data that actually needs to be written during the
 * compression or read during the decompression. However, the compression rates
 * obtained with this approach might be considered misleading, since it could
 * be argued that the unused gap should not be counted (but it needs to be
 * added to nvalue). A second boolean template parameter, pessimisticGap, lets
 * you decide how to handle this issue:
 * - false: There will be no gap between the selectors area and the data area
 *          (except for a small SIMD-padding). The reported compression rates
 *          will be correct. For the original variant, this does not cause any
 *          overhead. For the variant with the ring buffer, this requires to
 *          copy the whole data area directly behind the selectors area, which
 *          means a runtime overhead.
 * - true: There will be a pessimistic gap between the two areas. The reported
 *         compression rates will be misleading, unless we really have the
 *         worst case (each input group contains at least one value of more than
 *         16 bits). This causes no run time overhead, neither for the original
 *         nor for the variant using the ring buffer.
 * To sum up: For maximum performance use SIMDGroupSimple<false, false> or
 * SIMDGroupSimple<true, true>; to verify that the two variants really produce
 * the same data, use the same value for pessimisticGap.
 * 
 * Further assumptions
 * ===================
 * Finally, this implementation assumes that the number of 32-bit integers to
 * be compressed is a multiple of eight, so it should be used with
 * CompositeCodec.
 */
template<bool useRingBuf, bool pessimisticGap>
class SIMDGroupSimple256 : public IntegerCODEC {
public:
  // Tell CompositeCodec that this implementation can only handle input sizes
  // which are multiples of eight.
  static const uint32_t BlockSize = sizeof(__m256i) / sizeof(uint32_t);

  // The header consists of three 32-bit integers.
  static const uint32_t countHeader32 = 3;

  // Lookup table. Key: a selector, value: the number of groups to be packed
  // into one compressed block with the specified selector.
  static const uint8_t tableNum[];
  // Lookup table. Key: a selector, value: the mask required in the pattern
  // selection algorithm. Note that unlike in the paper, we look up the mask
  // directly instead of the bit width.
  static const uint32_t tableMask[];

  /**
   * Extracts the pos-th 4-bit selector from the selectors area, which starts
   * at inSelArea8. Note that, as described in the original paper, two
   * selectors are stored in each byte in the selectors area.
   */
  inline static uint8_t extractSel(const uint8_t *const &inSelArea8,
                                   const size_t &pos) {
    // We either need to extract the lower or the upper four bits of the
    // respective selector byte.
    return (pos & 1)
        ? ((inSelArea8[pos >> 1]) >> 4)
        : ((inSelArea8[pos >> 1]) & 0b1111);
  }

  /**
   * Utility function to calculate the number of padding bytes needed after the
   * selectors area in order to guarantee the 32-byte alignment required for
   * AVX2-store instructions in the data area.
   */
  inline static size_t getCountPadBytes(const size_t &countSelArea8) {
    const size_t offset = (countHeader32 * sizeof(uint32_t) +
        countSelArea8 + sizeof(uint8_t)) % sizeof(__m256i);
    return offset ? (sizeof(__m256i) - offset) : 0;
  }

  /**
   * This function is used to compress the n groups, i.e. 8x n integers, in the
   * last input block, if that last block is not "full". Note that this
   * function is called at most once per array to compress. Hence, top
   * efficiency is not that crucial here.
   */
  inline static void comprIncompleteBlock(const uint8_t &n, const __m256i *&in,
                                          __m256i *&out) {
    // Since we have to produce exactly one compressed vector anyway, we can
    // use the highest bit width allowing us to pack all n values.
    const unsigned b = 32 / n;
    __m256i comprBlock = _mm256_load_si256(in++);
    for (size_t k = 1; k < n; k++)
      comprBlock = _mm256_or_si256(comprBlock,
                                _mm256_slli_epi32(_mm256_load_si256(in++), k * b));
    _mm256_store_si256(out++, comprBlock);
  }

  /**
   * The following ten functions pack a certain amount of uncompressed data.
   * The function unrolledPacking_#n_#b packs #n groups, i.e., 8x #n integers,
   * into one 256-bit compressed block.
   */

  inline static __m256i unrolledPacking_32_1(const __m256i *&in) {
    __m256i res = _mm256_load_si256(in++);
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  1));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  2));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  3));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  4));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  5));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  6));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  7));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  8));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  9));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 10));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 11));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 12));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 13));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 14));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 15));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 16));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 17));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 18));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 19));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 20));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 21));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 22));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 23));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 24));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 25));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 26));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 27));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 28));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 29));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 30));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 31));
    return res;
  }

  inline static __m256i unrolledPacking_16_2(const __m256i *&in) {
    __m256i res = _mm256_load_si256(in++);
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  2));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  4));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  6));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  8));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 10));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 12));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 14));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 16));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 18));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 20));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 22));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 24));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 26));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 28));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 30));
    return res;
  }

  inline static __m256i unrolledPacking_10_3(const __m256i *&in) {
    __m256i res = _mm256_load_si256(in++);
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  3));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  6));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  9));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 12));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 15));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 18));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 21));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 24));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 27));
    return res;
  }

  inline static __m256i unrolledPacking_8_4(const __m256i *&in) {
    __m256i res = _mm256_load_si256(in++);
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  4));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  8));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 12));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 16));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 20));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 24));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 28));
    return res;
  }

  inline static __m256i unrolledPacking_6_5(const __m256i *&in) {
    __m256i res = _mm256_load_si256(in++);
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  5));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 10));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 15));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 20));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 25));
    return res;
  }

  inline static __m256i unrolledPacking_5_6(const __m256i *&in) {
    __m256i res = _mm256_load_si256(in++);
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  6));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 12));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 18));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 24));
    return res;
  }

  inline static __m256i unrolledPacking_4_8(const __m256i *&in) {
    __m256i res = _mm256_load_si256(in++);
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++),  8));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 16));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 24));
    return res;
  }

  inline static __m256i unrolledPacking_3_10(const __m256i *&in) {
    __m256i res = _mm256_load_si256(in++);
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 10));
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 20));
    return res;
  }

  inline static __m256i unrolledPacking_2_16(const __m256i *&in) {
    __m256i res = _mm256_load_si256(in++);
    res = _mm256_or_si256(res, _mm256_slli_epi32(_mm256_load_si256(in++), 16));
    return res;
  }

  inline static __m256i unrolledPacking_1_32(const __m256i *&in) {
    return _mm256_load_si256(in++);
  }

  /**
   * Compresses n groups, i.e. 8x n integers. Thereby, n must correspond to one
   * of the ten compression modes presented in the original paper.
   */
  inline static void comprCompleteBlock(const uint8_t &n, const __m256i *&in,
                                        __m256i *&out) {
    __m256i res;

    // In the following, b means the bit width.

    switch (n) {
      case 32: // b = 1
        res = unrolledPacking_32_1(in);
        break;
      case 16: // b = 2
        res = unrolledPacking_16_2(in);
        break;
      case 10: // b = 3
        res = unrolledPacking_10_3(in);
        break;
      case 8: // b = 4
        res = unrolledPacking_8_4(in);
        break;
      case 6: // b = 5
        res = unrolledPacking_6_5(in);
        break;
      case 5: // b = 6
        res = unrolledPacking_5_6(in);
        break;
      case 4: // b = 8
        res = unrolledPacking_4_8(in);
        break;
      case 3: // b = 10
        res = unrolledPacking_3_10(in);
        break;
      case 2: // b = 16
        res = unrolledPacking_2_16(in);
        break;
      case 1: // b = 32
        res = unrolledPacking_1_32(in);
        break;
    }

    _mm256_store_si256(out++, res);
  }

  /**
   * This function is used to decompress the n groups, i.e. 8x n integers, in
   * the last input block, if that last block is not "full". Note that this
   * function is called at most once per array to decompress. Hence, top
   * efficiency is not that crucial here.
   */
  inline static void decomprIncompleteBlock(const uint8_t &n, 
                                            const __m256i *&in,
                                            __m256i *&out) {
    // We choose the bit width consistent with comprIncompleteBlock().
    const unsigned b = 32 / n;
    const __m256i mask = _mm256_set1_epi32((static_cast<uint64_t>(1) << b) - 1);
    const __m256i comprBlock = _mm256_load_si256(in++);
    for (size_t k = 0; k < n; k++)
      _mm256_store_si256(out++,
                      _mm256_and_si256(_mm256_srli_epi32(comprBlock, k * b), mask));
  }

  /**
   * The following ten functions unpack a certain amount of compressed data.
   * The function unrolledUnpacking_#n_#b unpacks #n groups, i.e., 8x #n
   * integers, from one 256-bit compressed block.
   */

  inline static void unrolledUnpacking_32_1(const __m256i &comprBlock,
                                            __m256i *&out) {
    const __m256i mask = _mm256_set1_epi32(1);
    _mm256_store_si256(out++, _mm256_and_si256(                  comprBlock,      mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  1), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  2), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  3), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  4), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  5), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  6), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  7), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  8), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  9), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 10), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 11), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 12), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 13), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 14), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 15), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 16), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 17), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 18), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 19), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 20), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 21), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 22), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 23), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 24), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 25), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 26), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 27), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 28), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 29), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 30), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 31), mask));
  }

  inline static void unrolledUnpacking_16_2(const __m256i &comprBlock,
                                            __m256i *&out) {
    const __m256i mask = _mm256_set1_epi32((static_cast<uint32_t>(1) << 2) - 1);
    _mm256_store_si256(out++, _mm256_and_si256(                  comprBlock     , mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  2), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  4), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  6), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  8), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 10), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 12), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 14), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 16), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 18), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 20), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 22), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 24), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 26), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 28), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 30), mask));
  }

  inline static void unrolledUnpacking_10_3(const __m256i &comprBlock,
                                            __m256i *&out) {
    const __m256i mask = _mm256_set1_epi32((static_cast<uint32_t>(1) << 3) - 1);
    _mm256_store_si256(out++, _mm256_and_si256(                  comprBlock     , mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  3), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  6), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  9), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 12), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 15), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 18), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 21), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 24), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 27), mask));
  }

  inline static void unrolledUnpacking_8_4(const __m256i &comprBlock,
                                           __m256i *&out) {
    const __m256i mask = _mm256_set1_epi32((static_cast<uint32_t>(1) << 4) - 1);
    _mm256_store_si256(out++, _mm256_and_si256(                  comprBlock,      mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  4), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  8), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 12), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 16), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 20), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 24), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 28), mask));
  }

  inline static void unrolledUnpacking_6_5(const __m256i &comprBlock,
                                           __m256i *&out) {
    const __m256i mask = _mm256_set1_epi32((static_cast<uint32_t>(1) << 5) - 1);
    _mm256_store_si256(out++, _mm256_and_si256(                  comprBlock,      mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  5), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 10), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 15), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 20), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 25), mask));
  }

  inline static void unrolledUnpacking_5_6(const __m256i &comprBlock,
                                           __m256i *&out) {
    const __m256i mask = _mm256_set1_epi32((static_cast<uint32_t>(1) << 6) - 1);
    _mm256_store_si256(out++, _mm256_and_si256(                  comprBlock,      mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  6), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 12), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 18), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 24), mask));
  }

  inline static void unrolledUnpacking_4_8(const __m256i &comprBlock,
                                           __m256i *&out) {
    const __m256i mask = _mm256_set1_epi32((static_cast<uint32_t>(1) << 8) - 1);
    _mm256_store_si256(out++, _mm256_and_si256(                  comprBlock,      mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock,  8), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 16), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 24), mask));
  }

  inline static void unrolledUnpacking_3_10(const __m256i &comprBlock,
                                            __m256i *&out) {
    const __m256i mask = _mm256_set1_epi32((static_cast<uint32_t>(1) << 10) - 1);
    _mm256_store_si256(out++, _mm256_and_si256(                  comprBlock,      mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 10), mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 20), mask));
  }

  inline static void unrolledUnpacking_2_16(const __m256i &comprBlock,
                                            __m256i *&out) {
    const __m256i mask = _mm256_set1_epi32((static_cast<uint32_t>(1) << 16) - 1);
    _mm256_store_si256(out++, _mm256_and_si256(                  comprBlock,      mask));
    _mm256_store_si256(out++, _mm256_and_si256(_mm256_srli_epi32(comprBlock, 16), mask));
  }

  inline static void unrolledUnpacking_1_32(const __m256i &comprBlock,
                                            __m256i *&out) {
    _mm256_store_si256(out++, comprBlock);
  }

  /**
   * Decompresses n groups, i.e. 8x n integers. Thereby, n must correspond to
   * one of the ten compression modes presented in the original paper.
   */
  inline static void decomprCompleteBlock(const uint8_t &n,const __m256i *&in,
                                          __m256i *&out) {
    const __m256i comprBlock = _mm256_load_si256(in++);

    switch (n) {
      case 32: // b = 1
        unrolledUnpacking_32_1(comprBlock, out);
        break;
      case 16: // b = 2
        unrolledUnpacking_16_2(comprBlock, out);
        break;
      case 10: // b = 3
        unrolledUnpacking_10_3(comprBlock, out);
        break;
      case 8: // b = 4
        unrolledUnpacking_8_4(comprBlock, out);
        break;
      case 6: // b = 5
        unrolledUnpacking_6_5(comprBlock, out);
        break;
      case 5: // b = 6
        unrolledUnpacking_5_6(comprBlock, out);
        break;
      case 4: // b = 8
        unrolledUnpacking_4_8(comprBlock, out);
        break;
      case 3: // b = 10
        unrolledUnpacking_3_10(comprBlock, out);
        break;
      case 2: // b = 16
        unrolledUnpacking_2_16(comprBlock, out);
        break;
      case 1: // b = 32
        unrolledUnpacking_1_32(comprBlock, out);
        break;
    }
  }

  /**
   * The original variant of the compression part of the algorithm.
   */
  inline static void encodeArrayInternal_woRingBuf(const uint32_t *in,
                                                   const size_t len,
                                                   uint32_t *out,
                                                   size_t &nvalue) {
    // The start of the header.
    uint32_t *const outHeader32 = out;
    // The start of the selectors area.
    uint8_t *outSelArea8 =
        reinterpret_cast<uint8_t *>(outHeader32 + countHeader32);
    uint8_t *const initOutSelArea8 = outSelArea8;

    // The number of input groups, i.e., groups of eight integers. Note that we
    // assume the number of input integers to be a multiple of eight.
    const size_t countIn256 = len * sizeof(uint32_t) / sizeof(__m256i);

    // Step 1: Generation of the group max array
    // =========================================
    uint32_t *groupMaxArray = new uint32_t[countIn256];
    const unsigned countVectorElements = sizeof(__m256i) / sizeof(uint32_t);
    for (size_t i = 0; i < len; i += countVectorElements) {
      const uint32_t pseudoGroupMax =
          in[i     ] | in[i +  1] | in[i +  2] | in[i +  3] |
          in[i +  4] | in[i +  5] | in[i +  6] | in[i +  7]
      ;
      groupMaxArray[i / countVectorElements] = pseudoGroupMax;
    }

    // Step 2: Pattern selection algorithm
    // ===================================
    // As described in the paper.
    size_t l = countIn256;
    size_t j = 0;
    size_t pos = 0;
    // Whether we have an even number of selectors so far.
    bool even = true;
    while (l > 0) {
      uint8_t i;
      for (i = 0; i <= 9; i++) {
        const uint8_t n = tableNum[i];
        // Unlike the original pattern selection algorithm, we look up the mask
        // directly instead of calculating it from a looked up bit width.
        const uint32_t mask = tableMask[i];
        pos = 0;
        const size_t maxPos = min(static_cast<size_t>(n), l);
        while (pos < maxPos && groupMaxArray[j + pos] <= mask)
          pos++;
        if (pos == maxPos)
          break;
      }
      l -= pos;
      j += pos;
      // Store the selector.
      if (even)
        *outSelArea8 = i;
      else
        *outSelArea8++ |= (i << 4);
      even = !even;
    }
    if (!even)
      // The last used byte in the selectors area was touched, but not finished.
      outSelArea8++;
    // The number of groups in the last block.
    const uint8_t countGroupsLastBlock = static_cast<uint8_t>(pos);
    *outSelArea8 = countGroupsLastBlock;

    delete[] groupMaxArray;

    // The number of bytes actually used for the selectors area.
    const size_t countSelArea8Used = outSelArea8 - initOutSelArea8;
    // The total number of selectors.
    const int countSels = countSelArea8Used * 2 - (even ? 0 : 1);

    // The number of bytes that could be required for the selectors area in the
    // worst case.
    const size_t countSelArea8WorstCase = countIn256 / 2 + (countIn256 & 1);

    // Depending on whether we want to leave the "pessimistic gap" between the
    // selectors area and the data area, we either reserve the true or the
    // worst-case number of bytes for the selectors area. Note that this has no
    // effect on the amount of data that we actually have to write.
    const size_t countSelArea8 =
        pessimisticGap ? countSelArea8WorstCase : countSelArea8Used;

    const size_t countPadBytes = getCountPadBytes(countSelArea8);
    // The start of the data area.
    __m256i *outDataArea256 = reinterpret_cast<__m256i *>(initOutSelArea8 +
        countSelArea8 + sizeof(uint8_t) + countPadBytes);
    const __m256i *const initOutDataArea256 = outDataArea256;

    const __m256i *in256 = reinterpret_cast<const __m256i *>(in);

    // Step 3: Packing the uncompressed integers
    // =========================================
    // Iterate over the selectors obtained from the pattern selection algorithm
    // and compress the blocks accordingly. The last block is always treated
    // specially, since it might not be "full".
    for (int m = 0; m < countSels - 1; m++) {
      const uint8_t i = extractSel(initOutSelArea8, m);
      const size_t n = tableNum[i];
      comprCompleteBlock(n, in256, outDataArea256);
    }
    if (countGroupsLastBlock)
      comprIncompleteBlock(countGroupsLastBlock, in256, outDataArea256);

    // Write some meta data to the header.
    outHeader32[0] = len;
    outHeader32[1] = countSels;
    outHeader32[2] = countSelArea8;

    // The position of the last byte written to the output relative to the
    // start of the output. Note that the actual number of written bytes might
    // be slightly lower due to the inserted padding. However, it might even be
    // significantly lower, if pessimisticGap is true.
    const size_t nbytes = countHeader32 * sizeof(uint32_t) +
        countSelArea8 + sizeof(uint8_t) + countPadBytes +
        (outDataArea256 - initOutDataArea256) * sizeof(__m256i);
    // Rounding the number of bytes to full 32-bit integers.
    nvalue = div_roundup(nbytes, sizeof(uint32_t));
  }

  /**
   * The variant of the compression part using a ring buffer for the pseudo
   * group max values.
   */
  inline static void encodeArrayInternal_wRingBuf(const uint32_t *in,
                                                  const size_t len,
                                                  uint32_t *out,
                                                  size_t &nvalue) {
    // The start of the header.
    uint32_t *const outHeader32 = out;
    // The start of the selectors area.
    uint8_t *outSelArea8 =
        reinterpret_cast<uint8_t *>(outHeader32 + countHeader32);
    uint8_t *const initOutSelArea8 = outSelArea8;

    // The number of input groups, i.e., groups of eight integers. Note that we
    // assume the number of input integers to be a multiple of eight.
    const size_t countIn256 = len * sizeof(uint32_t) / sizeof(__m256i);

    // Maximum size of the group max ring buffer. Note that to determine the
    // next selector, we need to consider at most 32 pseudo group max values,
    // since that is the maximum number of input groups to be packed into one
    // compressed block.
    const size_t rbMaxSize = 32;
    // The group max ring buffer.
    uint32_t groupMaxRb[rbMaxSize];
    // The current position and number of valid elements in the ring buffer.
    size_t rbPos = 0;
    size_t rbSize = 0;

    // The number of bytes that could be required for the selectors area in the
    // worst case. In this implementation we immediately compress a block when
    // we have determined the selector. Hence, we do not know the total number
    // of selectors before we start the actual compression, such that we need
    // to assume the worst case in order to guarantee that the selectors area
    // and the data area do not overlap.
    const size_t countSelArea8WorstCase = countIn256 / 2 + (countIn256 & 1);
    size_t countPadBytes_wGap = getCountPadBytes(countSelArea8WorstCase);

    // The start of the data area.
    __m256i *outDataArea256_wGap =
        reinterpret_cast<__m256i *>(initOutSelArea8 + countSelArea8WorstCase +
        sizeof(uint8_t) + countPadBytes_wGap);
    __m256i *const initOutDataArea256_wGap = outDataArea256_wGap;

    const __m256i *in256 = reinterpret_cast<const __m256i *>(in);
    const __m256i *const endIn256 = in256 + countIn256;

    // The following loop interleaves all three steps of the original
    // algorithm: (1) the generation of the pseudo group max values, (2) the
    // pattern selection algorithm, and (3) the packing of the input blocks.

    // Whether we have an even number of selectors so far.
    bool even = true;
    size_t pos = 0;
    while (in256 < endIn256) {
      // Step 1: Refill the group max ring buffer.
      const size_t countRemainingIn256 = static_cast<size_t>(endIn256 - in256);
      const size_t rbSizeToReach = min(rbMaxSize, countRemainingIn256);
      for (; rbSize < rbSizeToReach; rbSize++) {
        const uint32_t *const in32 =
            reinterpret_cast<const uint32_t *>(in256 + rbSize);
        const uint32_t pseudoGroupMax =
            in32[0] | in32[1] | in32[2] | in32[3] |
            in32[4] | in32[5] | in32[6] | in32[7]
        ;
        groupMaxRb[(rbPos + rbSize) % rbMaxSize] = pseudoGroupMax;
      }

      // Step 2: Determine the next selector.
      pos = 0;
      uint8_t i;
      uint8_t n;
      for (i = 0; i <= 9; i++) {
        n = tableNum[i];
        const uint32_t mask = tableMask[i];
        pos = 0;
        const size_t maxPos = min(static_cast<size_t>(n), rbSize);
        while (pos < maxPos && groupMaxRb[(rbPos + pos) % rbMaxSize] <= mask)
          pos++;
        if (pos == maxPos)
          break;
      }
      // Store the selector.
      if (even)
        *outSelArea8 = i;
      else
        *outSelArea8++ |= (i << 4);
      even = !even;

      // Step 3: Compress the block.
      if (pos == n) {
        comprCompleteBlock(n, in256, outDataArea256_wGap);
        rbPos = (rbPos + n) % rbMaxSize;
        rbSize -= n;
        // Refilling the ring buffer only here (and once before the loop) does
        // not seem to yield any benefit.
      }
      else
        // This can only happen for the last block/selector
        comprIncompleteBlock(rbSize, in256, outDataArea256_wGap);
    }
    if (!even)
      // The last used byte in the selectors area was touched, but not finished.
      outSelArea8++;

    // The number of groups in the last, possibly non-"full" block.
    const uint8_t countGroupsLastBlock = static_cast<uint8_t>(pos);
    *outSelArea8 = countGroupsLastBlock;

    // The number of bytes actually used for the selectors area.
    const size_t countSelArea8Used = outSelArea8 - initOutSelArea8;
    // The total number of selectors.
    const size_t countSels = countSelArea8Used * 2 - (even ? 0 : 1);

    // Up to here, we have a gap between the last used byte of the selectors
    // area and the first byte of the data area (unless all data elements were
    // packed with 32 bits each, which is the worst case). If specified so, we
    // remove this gap by copying the data area directly behind the used bytes
    // of the selectors area.
    const size_t countSelArea8 =
        pessimisticGap ? countSelArea8WorstCase : countSelArea8Used;
    const size_t countDataArea256 =
        outDataArea256_wGap - initOutDataArea256_wGap;
    size_t actualPaddingBytes;
    if (pessimisticGap)
      actualPaddingBytes = countPadBytes_wGap;
    else {
      const size_t countPadBytes_woGap = getCountPadBytes(countSelArea8Used);
      actualPaddingBytes = countPadBytes_woGap;
      __m256i *const outDataArea256_woGap =
          reinterpret_cast<__m256i *>(initOutSelArea8 + countSelArea8Used +
          sizeof(uint8_t) + countPadBytes_woGap);
      if (outDataArea256_woGap != outDataArea256_wGap)
        for (unsigned i = 0; i < countDataArea256; i++)
          _mm256_store_si256(outDataArea256_woGap + i,
                          _mm256_load_si256(initOutDataArea256_wGap + i));
    }

    // Write some meta data to the header.
    outHeader32[0] = len;
    outHeader32[1] = countSels;
    outHeader32[2] = countSelArea8;

    // The position of the last byte written to the output relative to the
    // start of the output. Note that the actual number of written bytes might
    // be slightly lower due to the inserted padding. However, it might even be
    // significantly lower, if pessimisticGap is true.
    const size_t nbytes = countHeader32 * sizeof(uint32_t) +
        countSelArea8 + sizeof(uint8_t) + actualPaddingBytes +
        countDataArea256 * sizeof(__m256i);
    // Rounding the number of bytes to full 32-bit integers.
    nvalue = div_roundup(nbytes, sizeof(uint32_t));
  }

  void encodeArray(const uint32_t *in, const size_t len, uint32_t *out,
                   size_t &nvalue) {
    checkifdivisibleby(len, BlockSize);
    // TODO why doesn't this work? I manually inlined the function.
#if 0
    if (needPaddingTo256Bits(in))
#else
    if ((reinterpret_cast<uintptr_t>(in) & 31) != 0)
#endif
      throw std::runtime_error("the input buffer must be aligned to 32 bytes");
    
    if (useRingBuf)
      encodeArrayInternal_wRingBuf(in, len, out, nvalue);
    else
      encodeArrayInternal_woRingBuf(in, len, out, nvalue);
  }

  const uint32_t *decodeArray(const uint32_t *in, const size_t len,
                              uint32_t *out, size_t &nvalue) {
    // TODO why doesn't this work? I manually inlined the function.
#if 0
    if (needPaddingTo256Bits(out))
#else
    if ((reinterpret_cast<uintptr_t>(out) & 31) != 0)
#endif
      throw std::runtime_error("the output buffer must be aligned to 32 bytes");
    
    // The start of the header.
    const uint32_t *const inHeader32 = in;
    nvalue = inHeader32[0];
    const int countSels = inHeader32[1];
    // The number of bytes reserved for the selectors area. This contains the
    // bytes actually used for the selectors as well as the "pessimistic gap",
    // if specified so.
    const size_t countSelArea8 = inHeader32[2];

    // The start of the selectors area.
    const uint8_t *const inSelArea8 =
        reinterpret_cast<const uint8_t *>(inHeader32 + countHeader32);

    // The number of bytes actually used within the selectors area.
    const size_t countSelArea8Used = countSels / 2 + (countSels & 1);

    const size_t countPadBytes = getCountPadBytes(countSelArea8);
    // The start of the data area.
    const __m256i *inDataArea256 =
        reinterpret_cast<const __m256i *>(inSelArea8 + countSelArea8 +
        sizeof(uint8_t) + countPadBytes);

    __m256i *out256 = reinterpret_cast<__m256i *>(out);

    // Iterate over the selectors and unpack the compressed blocks accordingly.
    // The last block is always treated specially, since it might not be "full".
    for (int m = 0; m < countSels - 1; m++) {
      const uint8_t i = extractSel(inSelArea8, m);
      const size_t n = tableNum[i];
      decomprCompleteBlock(n, inDataArea256, out256);
    }
    const uint8_t countGroupsLastBlock = inSelArea8[countSelArea8Used];
    if (countGroupsLastBlock)
      decomprIncompleteBlock(countGroupsLastBlock, inDataArea256, out256);

    return reinterpret_cast<const uint32_t *>(inDataArea256);
  }

  virtual std::string name() const {
    std::ostringstream convert;
    convert << "SIMDGroupSimple256";
    if (useRingBuf)
      convert << "_RingBuf";
    return convert.str();
  }
};

template<bool useRingBuf, bool pessimisticGap>
const uint8_t SIMDGroupSimple256<useRingBuf, pessimisticGap>::tableNum[] = {
  32, 16, 10, 8, 6, 5, 4, 3, 2, 1
};
template<bool useRingBuf, bool pessimisticGap>
const uint32_t SIMDGroupSimple256<useRingBuf, pessimisticGap>::tableMask[] = {
  (static_cast<uint64_t>(1) <<  1) - 1,
  (static_cast<uint64_t>(1) <<  2) - 1,
  (static_cast<uint64_t>(1) <<  3) - 1,
  (static_cast<uint64_t>(1) <<  4) - 1,
  (static_cast<uint64_t>(1) <<  5) - 1,
  (static_cast<uint64_t>(1) <<  6) - 1,
  (static_cast<uint64_t>(1) <<  8) - 1,
  (static_cast<uint64_t>(1) << 10) - 1,
  (static_cast<uint64_t>(1) << 16) - 1,
  (static_cast<uint64_t>(1) << 32) - 1,
};

} // namespace FastPFor

#endif /* SIMDGROUPSIMPLE256_H_ */
