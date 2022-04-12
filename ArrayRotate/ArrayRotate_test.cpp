
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <immintrin.h>

#include <cstdint>
#include <cstddef>
#include <cstdbool>
#include <vector>
#include <atomic>
#include <algorithm>
#include <type_traits>

#if defined(__GNUC__) && !defined(__clang__)
#define JSTD_IS_PURE_GCC    1
#endif

//
// Since gcc 2.96 or Intel C++ compiler 8.0
//
// I'm not sure Intel C++ compiler 8.0 was the first version to support these builtins,
// update the condition if the version is not accurate. (Andrey Semashev)
//
#if (defined(__GNUC__) && ((__GNUC__ == 2 && __GNUC_MINOR__ >= 96) || (__GNUC__ >= 3))) \
    || (defined(__GNUC__) && (__INTEL_CXX_VERSION >= 800))
  #define SUPPORT_LIKELY        1
#elif defined(__clang__)
  //
  // clang: GCC extensions not implemented yet
  // See: http://clang.llvm.org/docs/UsersManual.html#gcc-extensions-not-implemented-yet
  //
  #if defined(__has_builtin)
    #if __has_builtin(__builtin_expect)
      #define SUPPORT_LIKELY    1
    #endif // __has_builtin(__builtin_expect)
  #endif // defined(__has_builtin)
#endif // SUPPORT_LIKELY

//
// Branch prediction hints
//
#if defined(SUPPORT_LIKELY) && (SUPPORT_LIKELY != 0)
#ifndef likely
#define likely(x)               __builtin_expect(!!(x), 1)
#endif
#ifndef unlikely
#define unlikely(x)             __builtin_expect(!!(x), 0)
#endif
#ifndef switch_likely
#define switch_likely(x, v)     __builtin_expect(!!(x), (v))
#endif
#else
#ifndef likely
#define likely(x)               (x)
#endif
#ifndef unlikely
#define unlikely(x)             (x)
#endif
#ifndef switch_likely
#define switch_likely(x, v)     (x)
#endif
#endif // likely() & unlikely()

/**
 * For inline, force-inline and no-inline define.
 */
#if defined(_MSC_VER)

#define JSTD_HAS_INLINE                     1

#define JSTD_INLINE                         __inline
#define JSTD_FORCE_INLINE                   __forceinline
#define JSTD_NO_INLINE                      __declspec(noinline)

#define JSTD_INLINE_DECLARE(type)           __inline type
#define JSTD_FORCE_INLINE_DECLARE(type)     __forceinline type
#define JSTD_NOINLINE_DECLARE(type)         __declspec(noinline) type

#define JSTD_CRT_INLINE                     extern __inline
#define JSTD_CRT_FORCE_INLINE               extern __forceinline
#define JSTD_CRT_NO_INLINE                  extern __declspec(noinline)

#define JSTD_CRT_INLINE_DECLARE(type)       extern __inline type
#define JSTD_CRT_FORCE_INLINE_DECLARE(type) extern __forceinline type
#define JSTD_CRT_NO_INLINE_DECLARE(type)    extern __declspec(noinline) type

#define JSTD_RESTRICT                       __restrict

#elif defined(__GNUC__) || defined(__clang__) || defined(__MINGW32__) || defined(__CYGWIN__)

#define JSTD_HAS_INLINE                     1

#define JSTD_INLINE                         inline __attribute__((gnu_inline))
#define JSTD_FORCE_INLINE                   inline __attribute__((always_inline))
#define JSTD_NO_INLINE                      __attribute__((noinline))

#define JSTD_INLINE_DECLARE(type)           inline __attribute__((gnu_inline)) type
#define JSTD_FORCE_INLINE_DECLARE(type)     inline __attribute__((always_inline)) type
#define JSTD_NOINLINE_DECLARE(type)         __attribute__((noinline)) type

#define JSTD_CRT_INLINE                     extern inline __attribute__((gnu_inline))
#define JSTD_CRT_FORCE_INLINE               extern inline __attribute__((always_inline))
#define JSTD_CRT_NO_INLINE                  extern __attribute__((noinline))

#define JSTD_CRT_INLINE_DECLARE(type)       extern inline __attribute__((gnu_inline)) type
#define JSTD_CRT_FORCE_INLINE_DECLARE(type) extern inline __attribute__((always_inline)) type
#define JSTD_CRT_NO_INLINE_DECLARE(type)    extern __attribute__((noinline)) type

#define JSTD_RESTRICT                       __restrict__

#else // Unknown compiler

#define JSTD_INLINE                         inline
#define JSTD_FORCE_INLINE                   inline
#define JSTD_NO_INLINE

#define JSTD_INLINE_DECLARE(type)           inline type
#define JSTD_FORCE_INLINE_DECLARE(type)     inline type
#define JSTD_NOINLINE_DECLARE(type)         type

#define JSTD_CRT_INLINE                     extern inline
#define JSTD_CRT_FORCE_INLINE               extern inline
#define JSTD_CRT_NO_INLINE                  extern

#define JSTD_CRT_INLINE_DECLARE(type)       extern inline type
#define JSTD_CRT_FORCE_INLINE_DECLARE(type) extern inline type
#define JSTD_CRT_NO_INLINE_DECLARE(type)    extern type

#define JSTD_RESTRICT

#endif // _MSC_VER

#ifndef JSTD_ASSERT
#ifdef _DEBUG
#define JSTD_ASSERT(express)            assert(!!(express))
#else
#define JSTD_ASSERT(express)            (void)0
#endif
#endif // JSTD_ASSERT

#ifndef JSTD_ASSERT_EX
#ifdef _DEBUG
#define JSTD_ASSERT_EX(express, text)   assert(!!(express))
#else
#define JSTD_ASSERT_EX(express, text)   (void)0
#endif
#endif // JSTD_ASSERT

#ifndef JSTD_STATIC_ASSERT
#if (__cplusplus < 201103L) || (defined(_MSC_VER) && (_MSC_VER < 1800))
#define JSTD_STATIC_ASSERT(express, text)       assert(!!(express))
#else
#define JSTD_STATIC_ASSERT(express, text)       static_assert(!!(express), text)
#endif
#endif // JSTD_STATIC_ASSERT

#ifndef PREFETCH_HINT_LEVEL
#define PREFETCH_HINT_LEVEL     _MM_HINT_T0
#endif

namespace jstd {

static const bool kUsePrefetchHint = true;
static const std::size_t kPrefetchOffset = 256;

#if defined(__GNUC__) && !defined(__clang__)
static const enum _mm_hint kPrefetchHintLevel = PREFETCH_HINT_LEVEL;
#else
static const int kPrefetchHintLevel = PREFETCH_HINT_LEVEL;
#endif

static const std::size_t kSSERegBytes = 16;
static const std::size_t kAVXRegBytes = 32;

static const std::size_t kSSERegCount = 8;
static const std::size_t kAVXRegCount = 16;

static const std::size_t kSSEAlignment = kSSERegBytes;
static const std::size_t kAVXAlignment = kAVXRegBytes;

static const std::size_t kSSEAlignMask = kSSEAlignment - 1;
static const std::size_t kAVXAlignMask = kAVXAlignment - 1;

static const std::size_t kRotateThresholdLength = 32;
static const std::size_t kMaxAVXStashBytes = (kAVXRegCount - 4) * kAVXRegBytes;

static const bool kSrcIsAligned = true;
static const bool kDestIsAligned = true;

static const bool kSrcIsNotAligned = false;
static const bool kDestIsNotAligned = false;

static const bool kLoadIsAligned = true;
static const bool kStoreIsAligned = true;

static const bool kLoadIsNotAligned = false;
static const bool kStoreIsNotAligned = false;

template <typename ForwardIter, typename ItemType = void>
ForwardIter // void until C++11
left_rotate(ForwardIter first, ForwardIter mid, ForwardIter last)
{
    typedef ForwardIter iterator;
    typedef typename std::iterator_traits<iterator>::difference_type    difference_type;
    typedef typename std::iterator_traits<iterator>::value_type         value_type;

    std::size_t left_len = (std::size_t)difference_type(mid - first);
    if (left_len == 0) return first;

    std::size_t right_len = (std::size_t)difference_type(last - mid);
    if (right_len == 0) return last;

    ForwardIter result = first + right_len;

    do {
        if (left_len <= right_len) {
            ForwardIter read = mid;
            ForwardIter write = first;
            if (left_len != 1) {
                while (read != last) {
                    std::iter_swap(write, read);
                    ++write;
                    ++read;
                }
                right_len %= left_len;
                first = write;
                left_len -= right_len;
                mid = last - right_len;
                if (right_len == 0 || left_len == 0)
                    break;
            }
            else {
                value_type tmp(std::move(*write));
                while (read != last) {
                    *write = *read;
                    ++write;
                    ++read;
                }
                *write = std::move(tmp);
                break;
            }
        }
        else {
            ForwardIter read = mid;
            ForwardIter write = last;
            if (right_len != 1) {
                while (read != first) {
                    --write;
                    --read;
                    std::iter_swap(read, write);
                }
                left_len %= right_len;
                last = write;
                right_len -= left_len;
                mid = first + left_len;
                if (left_len == 0 || right_len == 0)
                    break;
            }
            else {
                value_type tmp(std::move(*read));
                while (read != first) {
                    --write;
                    --read;
                    *write = *read;
                }
                *read = std::move(tmp);
                break;
            }
        }
    } while (1);

    return result;
}

#if 0

template <typename T, bool loadIsAligned, bool storeIsAligned, int LeftUints = 7>
static inline
void avx_forward_move_8_tailing(char * __restrict target, char * __restrict source, char * __restrict end)
{
    static const std::size_t kValueSize = sizeof(T);
    std::size_t lastUnalignedBytes = (std::size_t)end & kAVXAlignMask;
    const char * __restrict limit = end - lastUnalignedBytes;

    if (loadIsAligned && storeIsAligned) {
        if (((source + (8 * kAVXRegBytes)) <= limit) && (LeftUints >= 8)) {
            __m256i ymm0 = _mm256_load_si256((const __m256i *)(source + 32 * 0));
            __m256i ymm1 = _mm256_load_si256((const __m256i *)(source + 32 * 1));
            __m256i ymm2 = _mm256_load_si256((const __m256i *)(source + 32 * 2));
            __m256i ymm3 = _mm256_load_si256((const __m256i *)(source + 32 * 3));
            __m256i ymm4 = _mm256_load_si256((const __m256i *)(source + 32 * 4));
            __m256i ymm5 = _mm256_load_si256((const __m256i *)(source + 32 * 5));
            __m256i ymm6 = _mm256_load_si256((const __m256i *)(source + 32 * 6));
            __m256i ymm7 = _mm256_load_si256((const __m256i *)(source + 32 * 7));

            _mm256_store_si256((__m256i *)(target + 32 * 0), ymm0);
            _mm256_store_si256((__m256i *)(target + 32 * 1), ymm1);
            _mm256_store_si256((__m256i *)(target + 32 * 2), ymm2);
            _mm256_store_si256((__m256i *)(target + 32 * 3), ymm3);
            _mm256_store_si256((__m256i *)(target + 32 * 4), ymm4);
            _mm256_store_si256((__m256i *)(target + 32 * 5), ymm5);
            _mm256_store_si256((__m256i *)(target + 32 * 6), ymm6);
            _mm256_store_si256((__m256i *)(target + 32 * 7), ymm7);

            source += 8 * kAVXRegBytes;
            target += 8 * kAVXRegBytes;
        }

        if (((source + (4 * kAVXRegBytes)) <= limit) && (LeftUints >= 4)) {
            __m256i ymm0 = _mm256_load_si256((const __m256i *)(source + 32 * 0));
            __m256i ymm1 = _mm256_load_si256((const __m256i *)(source + 32 * 1));
            __m256i ymm2 = _mm256_load_si256((const __m256i *)(source + 32 * 2));
            __m256i ymm3 = _mm256_load_si256((const __m256i *)(source + 32 * 3));

            _mm256_store_si256((__m256i *)(target + 32 * 0), ymm0);
            _mm256_store_si256((__m256i *)(target + 32 * 1), ymm1);
            _mm256_store_si256((__m256i *)(target + 32 * 2), ymm2);
            _mm256_store_si256((__m256i *)(target + 32 * 3), ymm3);

            source += 4 * kAVXRegBytes;
            target += 4 * kAVXRegBytes;
        }

        if (((source + (2 * kAVXRegBytes)) <= limit) && (LeftUints >= 2)) {
            __m256i ymm0 = _mm256_load_si256((const __m256i *)(source + 32 * 0));
            __m256i ymm1 = _mm256_load_si256((const __m256i *)(source + 32 * 1));

            _mm256_store_si256((__m256i *)(target + 32 * 0), ymm0);
            _mm256_store_si256((__m256i *)(target + 32 * 1), ymm1);

            source += 2 * kAVXRegBytes;
            target += 2 * kAVXRegBytes;
        }

        if (((source + (1 * kAVXRegBytes)) <= limit) && (LeftUints >= 1)) {
            __m256i ymm0 = _mm256_load_si256((const __m256i *)(source + 32 * 0));

            _mm256_store_si256((__m256i *)(target + 32 * 0), ymm0);

            source += 1 * kAVXRegBytes;
            target += 1 * kAVXRegBytes;
        }
    }
    else if (loadIsAligned && !storeIsAligned) {
        if (((source + (8 * kAVXRegBytes)) <= limit) && (LeftUints >= 8)) {
            __m256i ymm0 = _mm256_load_si256((const __m256i *)(source + 32 * 0));
            __m256i ymm1 = _mm256_load_si256((const __m256i *)(source + 32 * 1));
            __m256i ymm2 = _mm256_load_si256((const __m256i *)(source + 32 * 2));
            __m256i ymm3 = _mm256_load_si256((const __m256i *)(source + 32 * 3));
            __m256i ymm4 = _mm256_load_si256((const __m256i *)(source + 32 * 4));
            __m256i ymm5 = _mm256_load_si256((const __m256i *)(source + 32 * 5));
            __m256i ymm6 = _mm256_load_si256((const __m256i *)(source + 32 * 6));
            __m256i ymm7 = _mm256_load_si256((const __m256i *)(source + 32 * 7));

            _mm256_storeu_si256((__m256i *)(target + 32 * 0), ymm0);
            _mm256_storeu_si256((__m256i *)(target + 32 * 1), ymm1);
            _mm256_storeu_si256((__m256i *)(target + 32 * 2), ymm2);
            _mm256_storeu_si256((__m256i *)(target + 32 * 3), ymm3);
            _mm256_storeu_si256((__m256i *)(target + 32 * 4), ymm4);
            _mm256_storeu_si256((__m256i *)(target + 32 * 5), ymm5);
            _mm256_storeu_si256((__m256i *)(target + 32 * 6), ymm6);
            _mm256_storeu_si256((__m256i *)(target + 32 * 7), ymm7);

            source += 8 * kAVXRegBytes;
            target += 8 * kAVXRegBytes;
        }

        if (((source + (4 * kAVXRegBytes)) <= limit) && (LeftUints >= 4)) {
            __m256i ymm0 = _mm256_load_si256((const __m256i *)(source + 32 * 0));
            __m256i ymm1 = _mm256_load_si256((const __m256i *)(source + 32 * 1));
            __m256i ymm2 = _mm256_load_si256((const __m256i *)(source + 32 * 2));
            __m256i ymm3 = _mm256_load_si256((const __m256i *)(source + 32 * 3));

            _mm256_storeu_si256((__m256i *)(target + 32 * 0), ymm0);
            _mm256_storeu_si256((__m256i *)(target + 32 * 1), ymm1);
            _mm256_storeu_si256((__m256i *)(target + 32 * 2), ymm2);
            _mm256_storeu_si256((__m256i *)(target + 32 * 3), ymm3);

            source += 4 * kAVXRegBytes;
            target += 4 * kAVXRegBytes;
        }

        if (((source + (2 * kAVXRegBytes)) <= limit) && (LeftUints >= 2)) {
            __m256i ymm0 = _mm256_load_si256((const __m256i *)(source + 32 * 0));
            __m256i ymm1 = _mm256_load_si256((const __m256i *)(source + 32 * 1));

            _mm256_storeu_si256((__m256i *)(target + 32 * 0), ymm0);
            _mm256_storeu_si256((__m256i *)(target + 32 * 1), ymm1);

            source += 2 * kAVXRegBytes;
            target += 2 * kAVXRegBytes;
        }

        if (((source + (1 * kAVXRegBytes)) <= limit) && (LeftUints >= 1)) {
            __m256i ymm0 = _mm256_load_si256((const __m256i *)(source + 32 * 0));

            _mm256_storeu_si256((__m256i *)(target + 32 * 0), ymm0);

            source += 1 * kAVXRegBytes;
            target += 1 * kAVXRegBytes;
        }
    }
    else if (!loadIsAligned && storeIsAligned) {
        if (((source + (8 * kAVXRegBytes)) <= limit) && (LeftUints >= 8)) {
            __m256i ymm0 = _mm256_loadu_si256((const __m256i *)(source + 32 * 0));
            __m256i ymm1 = _mm256_loadu_si256((const __m256i *)(source + 32 * 1));
            __m256i ymm2 = _mm256_loadu_si256((const __m256i *)(source + 32 * 2));
            __m256i ymm3 = _mm256_loadu_si256((const __m256i *)(source + 32 * 3));
            __m256i ymm4 = _mm256_loadu_si256((const __m256i *)(source + 32 * 4));
            __m256i ymm5 = _mm256_loadu_si256((const __m256i *)(source + 32 * 5));
            __m256i ymm6 = _mm256_loadu_si256((const __m256i *)(source + 32 * 6));
            __m256i ymm7 = _mm256_loadu_si256((const __m256i *)(source + 32 * 7));

            _mm256_store_si256((__m256i *)(target + 32 * 0), ymm0);
            _mm256_store_si256((__m256i *)(target + 32 * 1), ymm1);
            _mm256_store_si256((__m256i *)(target + 32 * 2), ymm2);
            _mm256_store_si256((__m256i *)(target + 32 * 3), ymm3);
            _mm256_store_si256((__m256i *)(target + 32 * 4), ymm4);
            _mm256_store_si256((__m256i *)(target + 32 * 5), ymm5);
            _mm256_store_si256((__m256i *)(target + 32 * 6), ymm6);
            _mm256_store_si256((__m256i *)(target + 32 * 7), ymm7);

            source += 8 * kAVXRegBytes;
            target += 8 * kAVXRegBytes;
        }

        if (((source + (4 * kAVXRegBytes)) <= limit) && (LeftUints >= 4)) {
            __m256i ymm0 = _mm256_loadu_si256((const __m256i *)(source + 32 * 0));
            __m256i ymm1 = _mm256_loadu_si256((const __m256i *)(source + 32 * 1));
            __m256i ymm2 = _mm256_loadu_si256((const __m256i *)(source + 32 * 2));
            __m256i ymm3 = _mm256_loadu_si256((const __m256i *)(source + 32 * 3));

            _mm256_store_si256((__m256i *)(target + 32 * 0), ymm0);
            _mm256_store_si256((__m256i *)(target + 32 * 1), ymm1);
            _mm256_store_si256((__m256i *)(target + 32 * 2), ymm2);
            _mm256_store_si256((__m256i *)(target + 32 * 3), ymm3);

            source += 4 * kAVXRegBytes;
            target += 4 * kAVXRegBytes;
        }

        if (((source + (2 * kAVXRegBytes)) <= limit) && (LeftUints >= 2)) {
            __m256i ymm0 = _mm256_loadu_si256((const __m256i *)(source + 32 * 0));
            __m256i ymm1 = _mm256_loadu_si256((const __m256i *)(source + 32 * 1));

            _mm256_store_si256((__m256i *)(target + 32 * 0), ymm0);
            _mm256_store_si256((__m256i *)(target + 32 * 1), ymm1);

            source += 2 * kAVXRegBytes;
            target += 2 * kAVXRegBytes;
        }

        if (((source + (1 * kAVXRegBytes)) <= limit) && (LeftUints >= 1)) {
            __m256i ymm0 = _mm256_loadu_si256((const __m256i *)(source + 32 * 0));

            _mm256_store_si256((__m256i *)(target + 32 * 0), ymm0);

            source += 1 * kAVXRegBytes;
            target += 1 * kAVXRegBytes;
        }
    }
    else {
        if (((source + (8 * kAVXRegBytes)) <= limit) && (LeftUints >= 8)) {
            __m256i ymm0 = _mm256_loadu_si256((const __m256i *)(source + 32 * 0));
            __m256i ymm1 = _mm256_loadu_si256((const __m256i *)(source + 32 * 1));
            __m256i ymm2 = _mm256_loadu_si256((const __m256i *)(source + 32 * 2));
            __m256i ymm3 = _mm256_loadu_si256((const __m256i *)(source + 32 * 3));
            __m256i ymm4 = _mm256_loadu_si256((const __m256i *)(source + 32 * 4));
            __m256i ymm5 = _mm256_loadu_si256((const __m256i *)(source + 32 * 5));
            __m256i ymm6 = _mm256_loadu_si256((const __m256i *)(source + 32 * 6));
            __m256i ymm7 = _mm256_loadu_si256((const __m256i *)(source + 32 * 7));

            _mm256_storeu_si256((__m256i *)(target + 32 * 0), ymm0);
            _mm256_storeu_si256((__m256i *)(target + 32 * 1), ymm1);
            _mm256_storeu_si256((__m256i *)(target + 32 * 2), ymm2);
            _mm256_storeu_si256((__m256i *)(target + 32 * 3), ymm3);
            _mm256_storeu_si256((__m256i *)(target + 32 * 4), ymm4);
            _mm256_storeu_si256((__m256i *)(target + 32 * 5), ymm5);
            _mm256_storeu_si256((__m256i *)(target + 32 * 6), ymm6);
            _mm256_storeu_si256((__m256i *)(target + 32 * 7), ymm7);

            source += 8 * kAVXRegBytes;
            target += 8 * kAVXRegBytes;
        }

        if (((source + (4 * kAVXRegBytes)) <= limit) && (LeftUints >= 4)) {
            __m256i ymm0 = _mm256_loadu_si256((const __m256i *)(source + 32 * 0));
            __m256i ymm1 = _mm256_loadu_si256((const __m256i *)(source + 32 * 1));
            __m256i ymm2 = _mm256_loadu_si256((const __m256i *)(source + 32 * 2));
            __m256i ymm3 = _mm256_loadu_si256((const __m256i *)(source + 32 * 3));

            _mm256_storeu_si256((__m256i *)(target + 32 * 0), ymm0);
            _mm256_storeu_si256((__m256i *)(target + 32 * 1), ymm1);
            _mm256_storeu_si256((__m256i *)(target + 32 * 2), ymm2);
            _mm256_storeu_si256((__m256i *)(target + 32 * 3), ymm3);

            source += 4 * kAVXRegBytes;
            target += 4 * kAVXRegBytes;
        }

        if (((source + (2 * kAVXRegBytes)) <= limit) && (LeftUints >= 2)) {
            __m256i ymm0 = _mm256_loadu_si256((const __m256i *)(source + 32 * 0));
            __m256i ymm1 = _mm256_loadu_si256((const __m256i *)(source + 32 * 1));

            _mm256_storeu_si256((__m256i *)(target + 32 * 0), ymm0);
            _mm256_storeu_si256((__m256i *)(target + 32 * 1), ymm1);

            source += 2 * kAVXRegBytes;
            target += 2 * kAVXRegBytes;
        }

        if (((source + (1 * kAVXRegBytes)) <= limit) && (LeftUints >= 1)) {
            __m256i ymm0 = _mm256_loadu_si256((const __m256i *)(source + 32 * 0));

            _mm256_storeu_si256((__m256i *)(target + 32 * 0), ymm0);

            source += 1 * kAVXRegBytes;
            target += 1 * kAVXRegBytes;
        }
    }

    while (source < end) {
        *(T *)target = *(T *)source;
        source += kValueSize;
        target += kValueSize;
    }
}

template <typename T, std::size_t N = 8>
static
void avx_forward_move_N_load_aligned(T * __restrict first, T * __restrict mid, T * __restrict last)
{
    static const std::size_t kValueSize = sizeof(T);
    static const bool kValueSizeIsPower2 = ((kValueSize & (kValueSize - 1)) == 0);
    static const bool kValueSizeIsDivisible =  (kValueSize < kAVXRegBytes) ?
                                              ((kAVXRegBytes % kValueSize) == 0) :
                                              ((kValueSize % kAVXRegBytes) == 0);
    // minimum AVX regs = 1, maximum AVX regs = 8
    static const std::size_t kAVXRegUnits = (N == 0) ? 1 : ((N <= 8) ? N : 8);
    static const std::size_t kPerStepBytes = kAVXRegUnits * kAVXRegBytes;

    std::size_t unAlignedBytes = (std::size_t)mid & kAVXAlignMask;
    bool loadAddrCanAlign;
    if (kValueSize < kAVXRegBytes)
        loadAddrCanAlign = (kValueSizeIsDivisible && ((unAlignedBytes % kValueSize) == 0));
    else
        loadAddrCanAlign = (kValueSizeIsDivisible && (unAlignedBytes == 0));

    if (likely(kValueSizeIsDivisible && loadAddrCanAlign)) {
        //unAlignedBytes = (unAlignedBytes != 0) ? (kAVXRegBytes - unAlignedBytes) : 0;
        unAlignedBytes = (kAVXRegBytes - unAlignedBytes) & kAVXAlignMask;
        while (unAlignedBytes != 0) {
            *first++ = *mid++;
            unAlignedBytes -= kValueSize;
        }

        char * __restrict target = (char * __restrict)first;
        char * __restrict source = (char * __restrict)mid;
        char * __restrict end = (char * __restrict)last;

        std::size_t lastUnalignedBytes = (std::size_t)last % kPerStepBytes;
        std::size_t totalBytes = (last - first) * kValueSize;
        const char * __restrict limit = (totalBytes >= kPerStepBytes) ? (end - lastUnalignedBytes) : source;

        bool storeAddrIsAligned = (((std::size_t)target & kAVXAlignMask) == 0);
        if (likely(!storeAddrIsAligned)) {
            while (source < limit) {
                //
                // See: https://blog.csdn.net/qq_43401808/article/details/87360789
                //
                if (kUsePrefetchHint) {
                    // Here, N would be best a multiple of 2.
                    _mm_prefetch((const char *)(source + kPrefetchOffset + 64 * 0), kPrefetchHintLevel);
                    if (N >= 3)
                    _mm_prefetch((const char *)(source + kPrefetchOffset + 64 * 1), kPrefetchHintLevel);
                    if (N >= 5)
                    _mm_prefetch((const char *)(source + kPrefetchOffset + 64 * 2), kPrefetchHintLevel);
                    if (N >= 7)
                    _mm_prefetch((const char *)(source + kPrefetchOffset + 64 * 3), kPrefetchHintLevel);
                }

                __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
                    ymm0 = _mm256_load_si256((const __m256i *)(source + 32 * 0));
                if (N >= 2)
                    ymm1 = _mm256_load_si256((const __m256i *)(source + 32 * 1));
                if (N >= 3)
                    ymm2 = _mm256_load_si256((const __m256i *)(source + 32 * 2));
                if (N >= 4)
                    ymm3 = _mm256_load_si256((const __m256i *)(source + 32 * 3));
                if (N >= 5)
                    ymm4 = _mm256_load_si256((const __m256i *)(source + 32 * 4));
                if (N >= 6)
                    ymm5 = _mm256_load_si256((const __m256i *)(source + 32 * 5));
                if (N >= 7)
                    ymm6 = _mm256_load_si256((const __m256i *)(source + 32 * 6));
                // Use "{" and "}" to avoid the gcc warnings
                if (N >= 8) {
                    ymm7 = _mm256_load_si256((const __m256i *)(source + 32 * 7));
                }

                    _mm256_storeu_si256((__m256i *)(target + 32 * 0), ymm0);
                if (N >= 2)
                    _mm256_storeu_si256((__m256i *)(target + 32 * 1), ymm1);
                if (N >= 3)
                    _mm256_storeu_si256((__m256i *)(target + 32 * 2), ymm2);
                if (N >= 4)
                    _mm256_storeu_si256((__m256i *)(target + 32 * 3), ymm3);
                if (N >= 5)
                    _mm256_storeu_si256((__m256i *)(target + 32 * 4), ymm4);
                if (N >= 6)
                    _mm256_storeu_si256((__m256i *)(target + 32 * 5), ymm5);
                if (N >= 7)
                    _mm256_storeu_si256((__m256i *)(target + 32 * 6), ymm6);
                // Use "{" and "}" to avoid the gcc warnings
                if (N >= 8) {
                    _mm256_storeu_si256((__m256i *)(target + 32 * 7), ymm7);
                }

                source += kPerStepBytes;
                target += kPerStepBytes;
            }

            avx_forward_move_8_tailing<T, kLoadIsAligned, kStoreIsNotAligned, N - 1>(target, source, end);
        } else {
            while (source < limit) {
                if (kUsePrefetchHint) {
                    // Here, N would be best a multiple of 2.
                    _mm_prefetch((const char *)(source + kPrefetchOffset + 64 * 0), kPrefetchHintLevel);
                    if (N >= 3)
                    _mm_prefetch((const char *)(source + kPrefetchOffset + 64 * 1), kPrefetchHintLevel);
                    if (N >= 5)
                    _mm_prefetch((const char *)(source + kPrefetchOffset + 64 * 2), kPrefetchHintLevel);
                    if (N >= 7)
                    _mm_prefetch((const char *)(source + kPrefetchOffset + 64 * 3), kPrefetchHintLevel);
                }

                __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
                    ymm0 = _mm256_load_si256((const __m256i *)(source + 32 * 0));
                if (N >= 2)
                    ymm1 = _mm256_load_si256((const __m256i *)(source + 32 * 1));
                if (N >= 3)
                    ymm2 = _mm256_load_si256((const __m256i *)(source + 32 * 2));
                if (N >= 4)
                    ymm3 = _mm256_load_si256((const __m256i *)(source + 32 * 3));
                if (N >= 5)
                    ymm4 = _mm256_load_si256((const __m256i *)(source + 32 * 4));
                if (N >= 6)
                    ymm5 = _mm256_load_si256((const __m256i *)(source + 32 * 5));
                if (N >= 7)
                    ymm6 = _mm256_load_si256((const __m256i *)(source + 32 * 6));
                // Use "{" and "}" to avoid the gcc warnings
                if (N >= 8) {
                    ymm7 = _mm256_load_si256((const __m256i *)(source + 32 * 7));
                }

                    _mm256_store_si256((__m256i *)(target + 32 * 0), ymm0);
                if (N >= 2)
                    _mm256_store_si256((__m256i *)(target + 32 * 1), ymm1);
                if (N >= 3)
                    _mm256_store_si256((__m256i *)(target + 32 * 2), ymm2);
                if (N >= 4)
                    _mm256_store_si256((__m256i *)(target + 32 * 3), ymm3);
                if (N >= 5)
                    _mm256_store_si256((__m256i *)(target + 32 * 4), ymm4);
                if (N >= 6)
                    _mm256_store_si256((__m256i *)(target + 32 * 5), ymm5);
                if (N >= 7)
                    _mm256_store_si256((__m256i *)(target + 32 * 6), ymm6);
                // Use "{" and "}" to avoid the gcc warnings
                if (N >= 8) {
                    _mm256_store_si256((__m256i *)(target + 32 * 7), ymm7);
                }

                source += kPerStepBytes;
                target += kPerStepBytes;
            }

            avx_forward_move_8_tailing<T, kLoadIsAligned, kStoreIsAligned, N - 1>(target, source, end);
        }
    }
    else {
        bool storeAddrCanAlign = false;
        if (kValueSizeIsDivisible) {
            unAlignedBytes = (std::size_t)first & kAVXAlignMask;
            if (kValueSize < kAVXRegBytes)
                storeAddrCanAlign = ((unAlignedBytes % kValueSize) == 0);
            else 
                storeAddrCanAlign = (unAlignedBytes == 0);

            if (storeAddrCanAlign) {
                unAlignedBytes = (kAVXRegBytes - unAlignedBytes) & kAVXAlignMask;
                while (unAlignedBytes != 0) {
                    *first++ = *mid++;
                    unAlignedBytes -= kValueSize;
                }
            }
        }

        char * __restrict target = (char * __restrict)first;
        char * __restrict source = (char * __restrict)mid;
        char * __restrict end = (char * __restrict)last;

        std::size_t lastUnalignedBytes = (std::size_t)last % kPerStepBytes;
        std::size_t totalBytes = (last - first) * kValueSize;
        const char * __restrict limit = (totalBytes >= kPerStepBytes) ? (end - lastUnalignedBytes) : source;

        if (likely(storeAddrCanAlign)) {
            while (source < limit) {
                if (kUsePrefetchHint) {
                    // Here, N would be best a multiple of 2.
                    _mm_prefetch((const char *)(source + kPrefetchOffset + 64 * 0), kPrefetchHintLevel);
                    if (N >= 3)
                    _mm_prefetch((const char *)(source + kPrefetchOffset + 64 * 1), kPrefetchHintLevel);
                    if (N >= 5)
                    _mm_prefetch((const char *)(source + kPrefetchOffset + 64 * 2), kPrefetchHintLevel);
                    if (N >= 7)
                    _mm_prefetch((const char *)(source + kPrefetchOffset + 64 * 3), kPrefetchHintLevel);
                }

                __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
                    ymm0 = _mm256_loadu_si256((const __m256i *)(source + 32 * 0));
                if (N >= 2)
                    ymm1 = _mm256_loadu_si256((const __m256i *)(source + 32 * 1));
                if (N >= 3)
                    ymm2 = _mm256_loadu_si256((const __m256i *)(source + 32 * 2));
                if (N >= 4)
                    ymm3 = _mm256_loadu_si256((const __m256i *)(source + 32 * 3));
                if (N >= 5)
                    ymm4 = _mm256_loadu_si256((const __m256i *)(source + 32 * 4));
                if (N >= 6)
                    ymm5 = _mm256_loadu_si256((const __m256i *)(source + 32 * 5));
                if (N >= 7)
                    ymm6 = _mm256_loadu_si256((const __m256i *)(source + 32 * 6));
                // Use "{" and "}" to avoid the gcc warnings
                if (N >= 8) {
                    ymm7 = _mm256_loadu_si256((const __m256i *)(source + 32 * 7));
                }

                    _mm256_store_si256((__m256i *)(target + 32 * 0), ymm0);
                if (N >= 2)
                    _mm256_store_si256((__m256i *)(target + 32 * 1), ymm1);
                if (N >= 3)
                    _mm256_store_si256((__m256i *)(target + 32 * 2), ymm2);
                if (N >= 4)
                    _mm256_store_si256((__m256i *)(target + 32 * 3), ymm3);
                if (N >= 5)
                    _mm256_store_si256((__m256i *)(target + 32 * 4), ymm4);
                if (N >= 6)
                    _mm256_store_si256((__m256i *)(target + 32 * 5), ymm5);
                if (N >= 7)
                    _mm256_store_si256((__m256i *)(target + 32 * 6), ymm6);
                // Use "{" and "}" to avoid the gcc warnings
                if (N >= 8) {
                    _mm256_store_si256((__m256i *)(target + 32 * 7), ymm7);
                }

                source += kPerStepBytes;
                target += kPerStepBytes;
            }

            avx_forward_move_8_tailing<T, kLoadIsNotAligned, kStoreIsAligned, N - 1>(target, source, end);
        } else {
            while (source < limit) {
                if (kUsePrefetchHint) {
                    // Here, N would be best a multiple of 2.
                    _mm_prefetch((const char *)(source + kPrefetchOffset + 64 * 0), kPrefetchHintLevel);
                    if (N >= 3)
                    _mm_prefetch((const char *)(source + kPrefetchOffset + 64 * 1), kPrefetchHintLevel);
                    if (N >= 5)
                    _mm_prefetch((const char *)(source + kPrefetchOffset + 64 * 2), kPrefetchHintLevel);
                    if (N >= 7)
                    _mm_prefetch((const char *)(source + kPrefetchOffset + 64 * 3), kPrefetchHintLevel);
                }

                __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
                    ymm0 = _mm256_loadu_si256((const __m256i *)(source + 32 * 0));
                if (N >= 2)
                    ymm1 = _mm256_loadu_si256((const __m256i *)(source + 32 * 1));
                if (N >= 3)
                    ymm2 = _mm256_loadu_si256((const __m256i *)(source + 32 * 2));
                if (N >= 4)
                    ymm3 = _mm256_loadu_si256((const __m256i *)(source + 32 * 3));
                if (N >= 5)
                    ymm4 = _mm256_loadu_si256((const __m256i *)(source + 32 * 4));
                if (N >= 6)
                    ymm5 = _mm256_loadu_si256((const __m256i *)(source + 32 * 5));
                if (N >= 7)
                    ymm6 = _mm256_loadu_si256((const __m256i *)(source + 32 * 6));
                // Use "{" and "}" to avoid the gcc warnings
                if (N >= 8) {
                    ymm7 = _mm256_loadu_si256((const __m256i *)(source + 32 * 7));
                }

                    _mm256_storeu_si256((__m256i *)(target + 32 * 0), ymm0);
                if (N >= 2)
                    _mm256_storeu_si256((__m256i *)(target + 32 * 1), ymm1);
                if (N >= 3)
                    _mm256_storeu_si256((__m256i *)(target + 32 * 2), ymm2);
                if (N >= 4)
                    _mm256_storeu_si256((__m256i *)(target + 32 * 3), ymm3);
                if (N >= 5)
                    _mm256_storeu_si256((__m256i *)(target + 32 * 4), ymm4);
                if (N >= 6)
                    _mm256_storeu_si256((__m256i *)(target + 32 * 5), ymm5);
                if (N >= 7)
                    _mm256_storeu_si256((__m256i *)(target + 32 * 6), ymm6);
                // Use "{" and "}" to avoid the gcc warnings
                if (N >= 8) {
                    _mm256_storeu_si256((__m256i *)(target + 32 * 7), ymm7);
                }

                source += kPerStepBytes;
                target += kPerStepBytes;
            }

            avx_forward_move_8_tailing<T, kLoadIsNotAligned, kStoreIsNotAligned, N - 1>(target, source, end);
        }
    }
}

#endif

template <typename T, bool srcIsAligned, bool destIsAligned, int LeftUints = 7>
static
JSTD_NO_INLINE
void avx_forward_move_N_tailing(char * JSTD_RESTRICT dest, char * JSTD_RESTRICT src, char * JSTD_RESTRICT end)
{
    static const std::size_t kValueSize = sizeof(T);
    JSTD_ASSERT(end >= src);
    std::size_t left_bytes = (std::size_t)(end - src);
    JSTD_ASSERT((left_bytes % kValueSize) == 0);

    if (srcIsAligned && destIsAligned) {
        if (((src + (8 * kAVXRegBytes)) <= end) && (LeftUints >= 8)) {
            __m256i ymm0 = _mm256_load_si256((const __m256i *)(src + 32 * 0));
            __m256i ymm1 = _mm256_load_si256((const __m256i *)(src + 32 * 1));
            __m256i ymm2 = _mm256_load_si256((const __m256i *)(src + 32 * 2));
            __m256i ymm3 = _mm256_load_si256((const __m256i *)(src + 32 * 3));
            __m256i ymm4 = _mm256_load_si256((const __m256i *)(src + 32 * 4));
            __m256i ymm5 = _mm256_load_si256((const __m256i *)(src + 32 * 5));
            __m256i ymm6 = _mm256_load_si256((const __m256i *)(src + 32 * 6));
            __m256i ymm7 = _mm256_load_si256((const __m256i *)(src + 32 * 7));

            src += 8 * kAVXRegBytes;

            _mm256_store_si256((__m256i *)(dest + 32 * 0), ymm0);
            _mm256_store_si256((__m256i *)(dest + 32 * 1), ymm1);
            _mm256_store_si256((__m256i *)(dest + 32 * 2), ymm2);
            _mm256_store_si256((__m256i *)(dest + 32 * 3), ymm3);
            _mm256_store_si256((__m256i *)(dest + 32 * 4), ymm4);
            _mm256_store_si256((__m256i *)(dest + 32 * 5), ymm5);
            _mm256_store_si256((__m256i *)(dest + 32 * 6), ymm6);
            _mm256_store_si256((__m256i *)(dest + 32 * 7), ymm7);

            dest += 8 * kAVXRegBytes;
        }

        if (((src + (4 * kAVXRegBytes)) <= end) && (LeftUints >= 4)) {
            __m256i ymm0 = _mm256_load_si256((const __m256i *)(src + 32 * 0));
            __m256i ymm1 = _mm256_load_si256((const __m256i *)(src + 32 * 1));
            __m256i ymm2 = _mm256_load_si256((const __m256i *)(src + 32 * 2));
            __m256i ymm3 = _mm256_load_si256((const __m256i *)(src + 32 * 3));

            src += 4 * kAVXRegBytes;

            _mm256_store_si256((__m256i *)(dest + 32 * 0), ymm0);
            _mm256_store_si256((__m256i *)(dest + 32 * 1), ymm1);
            _mm256_store_si256((__m256i *)(dest + 32 * 2), ymm2);
            _mm256_store_si256((__m256i *)(dest + 32 * 3), ymm3);

            dest += 4 * kAVXRegBytes;
        }

        if (((src + (2 * kAVXRegBytes)) <= end) && (LeftUints >= 2)) {
            __m256i ymm0 = _mm256_load_si256((const __m256i *)(src + 32 * 0));
            __m256i ymm1 = _mm256_load_si256((const __m256i *)(src + 32 * 1));

            src += 2 * kAVXRegBytes;

            _mm256_store_si256((__m256i *)(dest + 32 * 0), ymm0);
            _mm256_store_si256((__m256i *)(dest + 32 * 1), ymm1);

            dest += 2 * kAVXRegBytes;
        }

        if (((src + (1 * kAVXRegBytes)) <= end) && (LeftUints >= 1)) {
            __m256i ymm0 = _mm256_load_si256((const __m256i *)(src + 32 * 0));

            src += 1 * kAVXRegBytes;

            _mm256_store_si256((__m256i *)(dest + 32 * 0), ymm0);

            dest += 1 * kAVXRegBytes;
        }
    }
    else if (srcIsAligned && !destIsAligned) {
        if (((src + (8 * kAVXRegBytes)) <= end) && (LeftUints >= 8)) {
            __m256i ymm0 = _mm256_load_si256((const __m256i *)(src + 32 * 0));
            __m256i ymm1 = _mm256_load_si256((const __m256i *)(src + 32 * 1));
            __m256i ymm2 = _mm256_load_si256((const __m256i *)(src + 32 * 2));
            __m256i ymm3 = _mm256_load_si256((const __m256i *)(src + 32 * 3));
            __m256i ymm4 = _mm256_load_si256((const __m256i *)(src + 32 * 4));
            __m256i ymm5 = _mm256_load_si256((const __m256i *)(src + 32 * 5));
            __m256i ymm6 = _mm256_load_si256((const __m256i *)(src + 32 * 6));
            __m256i ymm7 = _mm256_load_si256((const __m256i *)(src + 32 * 7));

            src += 8 * kAVXRegBytes;

            _mm256_storeu_si256((__m256i *)(dest + 32 * 0), ymm0);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 1), ymm1);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 2), ymm2);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 3), ymm3);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 4), ymm4);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 5), ymm5);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 6), ymm6);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 7), ymm7);

            dest += 8 * kAVXRegBytes;
        }

        if (((src + (4 * kAVXRegBytes)) <= end) && (LeftUints >= 4)) {
            __m256i ymm0 = _mm256_load_si256((const __m256i *)(src + 32 * 0));
            __m256i ymm1 = _mm256_load_si256((const __m256i *)(src + 32 * 1));
            __m256i ymm2 = _mm256_load_si256((const __m256i *)(src + 32 * 2));
            __m256i ymm3 = _mm256_load_si256((const __m256i *)(src + 32 * 3));

            src += 4 * kAVXRegBytes;

            _mm256_storeu_si256((__m256i *)(dest + 32 * 0), ymm0);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 1), ymm1);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 2), ymm2);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 3), ymm3);

            dest += 4 * kAVXRegBytes;
        }

        if (((src + (2 * kAVXRegBytes)) <= end) && (LeftUints >= 2)) {
            __m256i ymm0 = _mm256_load_si256((const __m256i *)(src + 32 * 0));
            __m256i ymm1 = _mm256_load_si256((const __m256i *)(src + 32 * 1));

            src += 2 * kAVXRegBytes;

            _mm256_storeu_si256((__m256i *)(dest + 32 * 0), ymm0);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 1), ymm1);

            dest += 2 * kAVXRegBytes;
        }

        if (((src + (1 * kAVXRegBytes)) <= end) && (LeftUints >= 1)) {
            __m256i ymm0 = _mm256_load_si256((const __m256i *)(src + 32 * 0));

            src += 1 * kAVXRegBytes;

            _mm256_storeu_si256((__m256i *)(dest + 32 * 0), ymm0);

            dest += 1 * kAVXRegBytes;
        }
    }
    else if (!srcIsAligned && destIsAligned) {
        if (((src + (8 * kAVXRegBytes)) <= end) && (LeftUints >= 8)) {
            __m256i ymm0 = _mm256_loadu_si256((const __m256i *)(src + 32 * 0));
            __m256i ymm1 = _mm256_loadu_si256((const __m256i *)(src + 32 * 1));
            __m256i ymm2 = _mm256_loadu_si256((const __m256i *)(src + 32 * 2));
            __m256i ymm3 = _mm256_loadu_si256((const __m256i *)(src + 32 * 3));
            __m256i ymm4 = _mm256_loadu_si256((const __m256i *)(src + 32 * 4));
            __m256i ymm5 = _mm256_loadu_si256((const __m256i *)(src + 32 * 5));
            __m256i ymm6 = _mm256_loadu_si256((const __m256i *)(src + 32 * 6));
            __m256i ymm7 = _mm256_loadu_si256((const __m256i *)(src + 32 * 7));

            src += 8 * kAVXRegBytes;

            _mm256_store_si256((__m256i *)(dest + 32 * 0), ymm0);
            _mm256_store_si256((__m256i *)(dest + 32 * 1), ymm1);
            _mm256_store_si256((__m256i *)(dest + 32 * 2), ymm2);
            _mm256_store_si256((__m256i *)(dest + 32 * 3), ymm3);
            _mm256_store_si256((__m256i *)(dest + 32 * 4), ymm4);
            _mm256_store_si256((__m256i *)(dest + 32 * 5), ymm5);
            _mm256_store_si256((__m256i *)(dest + 32 * 6), ymm6);
            _mm256_store_si256((__m256i *)(dest + 32 * 7), ymm7);

            dest += 8 * kAVXRegBytes;
        }

        if (((src + (4 * kAVXRegBytes)) <= end) && (LeftUints >= 4)) {
            __m256i ymm0 = _mm256_loadu_si256((const __m256i *)(src + 32 * 0));
            __m256i ymm1 = _mm256_loadu_si256((const __m256i *)(src + 32 * 1));
            __m256i ymm2 = _mm256_loadu_si256((const __m256i *)(src + 32 * 2));
            __m256i ymm3 = _mm256_loadu_si256((const __m256i *)(src + 32 * 3));

            src += 4 * kAVXRegBytes;

            _mm256_store_si256((__m256i *)(dest + 32 * 0), ymm0);
            _mm256_store_si256((__m256i *)(dest + 32 * 1), ymm1);
            _mm256_store_si256((__m256i *)(dest + 32 * 2), ymm2);
            _mm256_store_si256((__m256i *)(dest + 32 * 3), ymm3);

            dest += 4 * kAVXRegBytes;
        }

        if (((src + (2 * kAVXRegBytes)) <= end) && (LeftUints >= 2)) {
            __m256i ymm0 = _mm256_loadu_si256((const __m256i *)(src + 32 * 0));
            __m256i ymm1 = _mm256_loadu_si256((const __m256i *)(src + 32 * 1));

            src += 2 * kAVXRegBytes;

            _mm256_store_si256((__m256i *)(dest + 32 * 0), ymm0);
            _mm256_store_si256((__m256i *)(dest + 32 * 1), ymm1);

            dest += 2 * kAVXRegBytes;
        }

        if (((src + (1 * kAVXRegBytes)) <= end) && (LeftUints >= 1)) {
            __m256i ymm0 = _mm256_loadu_si256((const __m256i *)(src + 32 * 0));

            src += 1 * kAVXRegBytes;

            _mm256_store_si256((__m256i *)(dest + 32 * 0), ymm0);

            dest += 1 * kAVXRegBytes;
        }
    }
    else {
        if (((src + (8 * kAVXRegBytes)) <= end) && (LeftUints >= 8)) {
            __m256i ymm0 = _mm256_loadu_si256((const __m256i *)(src + 32 * 0));
            __m256i ymm1 = _mm256_loadu_si256((const __m256i *)(src + 32 * 1));
            __m256i ymm2 = _mm256_loadu_si256((const __m256i *)(src + 32 * 2));
            __m256i ymm3 = _mm256_loadu_si256((const __m256i *)(src + 32 * 3));
            __m256i ymm4 = _mm256_loadu_si256((const __m256i *)(src + 32 * 4));
            __m256i ymm5 = _mm256_loadu_si256((const __m256i *)(src + 32 * 5));
            __m256i ymm6 = _mm256_loadu_si256((const __m256i *)(src + 32 * 6));
            __m256i ymm7 = _mm256_loadu_si256((const __m256i *)(src + 32 * 7));

            src += 8 * kAVXRegBytes;

            _mm256_storeu_si256((__m256i *)(dest + 32 * 0), ymm0);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 1), ymm1);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 2), ymm2);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 3), ymm3);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 4), ymm4);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 5), ymm5);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 6), ymm6);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 7), ymm7);

            dest += 8 * kAVXRegBytes;
        }

        if (((src + (4 * kAVXRegBytes)) <= end) && (LeftUints >= 4)) {
            __m256i ymm0 = _mm256_loadu_si256((const __m256i *)(src + 32 * 0));
            __m256i ymm1 = _mm256_loadu_si256((const __m256i *)(src + 32 * 1));
            __m256i ymm2 = _mm256_loadu_si256((const __m256i *)(src + 32 * 2));
            __m256i ymm3 = _mm256_loadu_si256((const __m256i *)(src + 32 * 3));

            src += 4 * kAVXRegBytes;

            _mm256_storeu_si256((__m256i *)(dest + 32 * 0), ymm0);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 1), ymm1);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 2), ymm2);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 3), ymm3);

            dest += 4 * kAVXRegBytes;
        }

        if (((src + (2 * kAVXRegBytes)) <= end) && (LeftUints >= 2)) {
            __m256i ymm0 = _mm256_loadu_si256((const __m256i *)(src + 32 * 0));
            __m256i ymm1 = _mm256_loadu_si256((const __m256i *)(src + 32 * 1));

            src += 2 * kAVXRegBytes;

            _mm256_storeu_si256((__m256i *)(dest + 32 * 0), ymm0);
            _mm256_storeu_si256((__m256i *)(dest + 32 * 1), ymm1);

            dest += 2 * kAVXRegBytes;
        }

        if (((src + (1 * kAVXRegBytes)) <= end) && (LeftUints >= 1)) {
            __m256i ymm0 = _mm256_loadu_si256((const __m256i *)(src + 32 * 0));

            src += 1 * kAVXRegBytes;

            _mm256_storeu_si256((__m256i *)(dest + 32 * 0), ymm0);

            dest += 1 * kAVXRegBytes;
        }
    }

    while (src < end) {
        *(T *)dest = *(T *)src;
        src += kValueSize;
        dest += kValueSize;
    }
}

//
// Code block alignment setting to 64 bytes maybe better than 32,
// but it's too wasteful for code size.
//
#if defined(JSTD_IS_PURE_GCC)
#pragma GCC push_options
#pragma GCC optimize ("align-labels=32")
#endif

template <typename T, std::size_t N = 8, std::size_t estimatedSize = sizeof(T)>
static
JSTD_NO_INLINE
void avx_forward_move_N_store_aligned(T * JSTD_RESTRICT first, T * JSTD_RESTRICT mid, T * JSTD_RESTRICT last)
{
    static const std::size_t kValueSize = sizeof(T);
    static const bool kValueSizeIsPower2 = ((kValueSize & (kValueSize - 1)) == 0);
    static const bool kValueSizeIsDivisible =  (kValueSize < kAVXRegBytes) ?
                                              ((kAVXRegBytes % kValueSize) == 0) :
                                              ((kValueSize % kAVXRegBytes) == 0);
    // minimum AVX regs = 1, maximum AVX regs = 8
    static const std::size_t _N = (N == 0) ? 1 : ((N <= 8) ? N : 8);
    static const std::size_t kSingleLoopBytes = _N * kAVXRegBytes;

    std::size_t unAlignedBytes = (std::size_t)first & kAVXAlignMask;
    bool destAddrCanAlign;
    if (kValueSize < kAVXRegBytes)
        destAddrCanAlign = (kValueSizeIsDivisible && ((unAlignedBytes % kValueSize) == 0));
    else
        destAddrCanAlign = (kValueSizeIsDivisible && (unAlignedBytes == 0));

    if (likely(kValueSizeIsDivisible && destAddrCanAlign)) {
        unAlignedBytes = (kAVXRegBytes - unAlignedBytes) & kAVXAlignMask;
        while (unAlignedBytes != 0) {
            *first++ = *mid++;
            unAlignedBytes -= kValueSize;
        }

        char * JSTD_RESTRICT dest = (char * JSTD_RESTRICT)first;
        char * JSTD_RESTRICT src = (char * JSTD_RESTRICT)mid;
        char * JSTD_RESTRICT end = (char * JSTD_RESTRICT)last;

        std::size_t totalMoveBytes = (last - mid) * kValueSize;
        std::size_t unalignedMoveBytes = (std::size_t)totalMoveBytes % kSingleLoopBytes;
        const char * JSTD_RESTRICT limit = (totalMoveBytes >= kSingleLoopBytes) ? (end - unalignedMoveBytes) : src;

        bool srcAddrIsAligned = (((std::size_t)src & kAVXAlignMask) == 0);
        if (likely(!srcAddrIsAligned)) {
#if defined(JSTD_IS_ICC)
#pragma code_align(64)
#endif
            while (src < limit) {
                __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
                if (N >= 0) {
                    ymm0 = _mm256_loadu_si256((const __m256i *)(src + 32 * 0));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 2) {
                    ymm1 = _mm256_loadu_si256((const __m256i *)(src + 32 * 1));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 3) {
                    ymm2 = _mm256_loadu_si256((const __m256i *)(src + 32 * 2));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 4) {
                    ymm3 = _mm256_loadu_si256((const __m256i *)(src + 32 * 3));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 5) {
                    ymm4 = _mm256_loadu_si256((const __m256i *)(src + 32 * 4));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 6) {
                    ymm5 = _mm256_loadu_si256((const __m256i *)(src + 32 * 5));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 7) {
                    ymm6 = _mm256_loadu_si256((const __m256i *)(src + 32 * 6));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                // Use "{" and "}" to avoid the gcc warnings
                if (N >= 8) {
                    ymm7 = _mm256_loadu_si256((const __m256i *)(src + 32 * 7));
                    std::atomic_signal_fence(std::memory_order_release);
                }

                //
                // See: https://blog.csdn.net/qq_43401808/article/details/87360789
                //
                if (kUsePrefetchHint) {
                    // Here, N would be best a multiple of 2.
                    _mm_prefetch((const char *)(src + kPrefetchOffset + 64 * 0), kPrefetchHintLevel);
                    if (N >= 3)
                    _mm_prefetch((const char *)(src + kPrefetchOffset + 64 * 1), kPrefetchHintLevel);
                    if (N >= 5)
                    _mm_prefetch((const char *)(src + kPrefetchOffset + 64 * 2), kPrefetchHintLevel);
                    if (N >= 7)
                    _mm_prefetch((const char *)(src + kPrefetchOffset + 64 * 3), kPrefetchHintLevel);
                }

                src += kSingleLoopBytes;

                if (N >= 0) {
                    _mm256_store_si256((__m256i *)(dest + 32 * 0), ymm0);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 2) {
                    _mm256_store_si256((__m256i *)(dest + 32 * 1), ymm1);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 3) {
                    _mm256_store_si256((__m256i *)(dest + 32 * 2), ymm2);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 4) {
                    _mm256_store_si256((__m256i *)(dest + 32 * 3), ymm3);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 5) {
                    _mm256_store_si256((__m256i *)(dest + 32 * 4), ymm4);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 6) {
                    _mm256_store_si256((__m256i *)(dest + 32 * 5), ymm5);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 7) {
                    _mm256_store_si256((__m256i *)(dest + 32 * 6), ymm6);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                // Use "{" and "}" to avoid the gcc warnings
                if (N >= 8) {
                    _mm256_store_si256((__m256i *)(dest + 32 * 7), ymm7);
                    std::atomic_signal_fence(std::memory_order_release);
                }

                dest += kSingleLoopBytes;
            }

            avx_forward_move_N_tailing<T, kSrcIsNotAligned, kDestIsAligned, _N - 1>(dest, src, end);
        } else {
#if defined(JSTD_IS_ICC)
#pragma code_align(64)
#endif
            while (src < limit) {
                __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
                if (N >= 0) {
                    ymm0 = _mm256_load_si256((const __m256i *)(src + 32 * 0));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 2) {
                    ymm1 = _mm256_load_si256((const __m256i *)(src + 32 * 1));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 3) {
                    ymm2 = _mm256_load_si256((const __m256i *)(src + 32 * 2));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 4) {
                    ymm3 = _mm256_load_si256((const __m256i *)(src + 32 * 3));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 5) {
                    ymm4 = _mm256_load_si256((const __m256i *)(src + 32 * 4));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 6) {
                    ymm5 = _mm256_load_si256((const __m256i *)(src + 32 * 5));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 7) {
                    ymm6 = _mm256_load_si256((const __m256i *)(src + 32 * 6));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                // Use "{" and "}" to avoid the gcc warnings
                if (N >= 8) {
                    ymm7 = _mm256_load_si256((const __m256i *)(src + 32 * 7));
                    std::atomic_signal_fence(std::memory_order_release);
                }

                if (kUsePrefetchHint) {
                    // Here, N would be best a multiple of 2.
                    _mm_prefetch((const char *)(src + kPrefetchOffset + 64 * 0), kPrefetchHintLevel);
                    if (N >= 3)
                    _mm_prefetch((const char *)(src + kPrefetchOffset + 64 * 1), kPrefetchHintLevel);
                    if (N >= 5)
                    _mm_prefetch((const char *)(src + kPrefetchOffset + 64 * 2), kPrefetchHintLevel);
                    if (N >= 7)
                    _mm_prefetch((const char *)(src + kPrefetchOffset + 64 * 3), kPrefetchHintLevel);
                }

                src += kSingleLoopBytes;

                if (N >= 0) {
                    _mm256_store_si256((__m256i *)(dest + 32 * 0), ymm0);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 2) {
                    _mm256_store_si256((__m256i *)(dest + 32 * 1), ymm1);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 3) {
                    _mm256_store_si256((__m256i *)(dest + 32 * 2), ymm2);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 4) {
                    _mm256_store_si256((__m256i *)(dest + 32 * 3), ymm3);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 5) {
                    _mm256_store_si256((__m256i *)(dest + 32 * 4), ymm4);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 6) {
                    _mm256_store_si256((__m256i *)(dest + 32 * 5), ymm5);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 7) {
                    _mm256_store_si256((__m256i *)(dest + 32 * 6), ymm6);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                // Use "{" and "}" to avoid the gcc warnings
                if (N >= 8) {
                    _mm256_store_si256((__m256i *)(dest + 32 * 7), ymm7);
                    std::atomic_signal_fence(std::memory_order_release);
                }

                dest += kSingleLoopBytes;
            }

            avx_forward_move_N_tailing<T, kSrcIsAligned, kDestIsAligned, _N - 1>(dest, src, end);
        }
    }
    else {
        bool srcAddrCanAlign = false;
        if (kValueSizeIsDivisible) {
            unAlignedBytes = (std::size_t)mid & kAVXAlignMask;
            if (kValueSize < kAVXRegBytes)
                srcAddrCanAlign = ((unAlignedBytes % kValueSize) == 0);
            else
                srcAddrCanAlign = (unAlignedBytes == 0);

            if (srcAddrCanAlign) {
                unAlignedBytes = (kAVXRegBytes - unAlignedBytes) & kAVXAlignMask;
                while (unAlignedBytes != 0) {
                    *first++ = *mid++;
                    unAlignedBytes -= kValueSize;
                }
            }
        }

        char * JSTD_RESTRICT dest = (char * JSTD_RESTRICT)first;
        char * JSTD_RESTRICT src = (char * JSTD_RESTRICT)mid;
        char * JSTD_RESTRICT end = (char * JSTD_RESTRICT)last;

        std::size_t totalMoveBytes = (last - mid) * kValueSize;
        std::size_t unalignedMoveBytes = (std::size_t)totalMoveBytes % kSingleLoopBytes;
        const char * JSTD_RESTRICT limit = (totalMoveBytes >= kSingleLoopBytes) ? (end - unalignedMoveBytes) : src;

        if (likely(srcAddrCanAlign)) {
#if defined(JSTD_IS_ICC)
#pragma code_align(64)
#endif
            while (src < limit) {
                __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
                if (N >= 0) {
                    ymm0 = _mm256_load_si256((const __m256i *)(src + 32 * 0));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 2) {
                    ymm1 = _mm256_load_si256((const __m256i *)(src + 32 * 1));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 3) {
                    ymm2 = _mm256_load_si256((const __m256i *)(src + 32 * 2));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 4) {
                    ymm3 = _mm256_load_si256((const __m256i *)(src + 32 * 3));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 5) {
                    ymm4 = _mm256_load_si256((const __m256i *)(src + 32 * 4));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 6) {
                    ymm5 = _mm256_load_si256((const __m256i *)(src + 32 * 5));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 7) {
                    ymm6 = _mm256_load_si256((const __m256i *)(src + 32 * 6));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                // Use "{" and "}" to avoid the gcc warnings
                if (N >= 8) {
                    ymm7 = _mm256_load_si256((const __m256i *)(src + 32 * 7));
                    std::atomic_signal_fence(std::memory_order_release);
                }

                if (kUsePrefetchHint) {
                    // Here, N would be best a multiple of 2.
                    _mm_prefetch((const char *)(src + kPrefetchOffset + 64 * 0), kPrefetchHintLevel);
                    if (N >= 3)
                    _mm_prefetch((const char *)(src + kPrefetchOffset + 64 * 1), kPrefetchHintLevel);
                    if (N >= 5)
                    _mm_prefetch((const char *)(src + kPrefetchOffset + 64 * 2), kPrefetchHintLevel);
                    if (N >= 7)
                    _mm_prefetch((const char *)(src + kPrefetchOffset + 64 * 3), kPrefetchHintLevel);
                }

                src += kSingleLoopBytes;

                if (N >= 0) {
                    _mm256_storeu_si256((__m256i *)(dest + 32 * 0), ymm0);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 2) {
                    _mm256_storeu_si256((__m256i *)(dest + 32 * 1), ymm1);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 3) {
                    _mm256_storeu_si256((__m256i *)(dest + 32 * 2), ymm2);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 4) {
                    _mm256_storeu_si256((__m256i *)(dest + 32 * 3), ymm3);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 5) {
                    _mm256_storeu_si256((__m256i *)(dest + 32 * 4), ymm4);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 6) {
                    _mm256_storeu_si256((__m256i *)(dest + 32 * 5), ymm5);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 7) {
                    _mm256_storeu_si256((__m256i *)(dest + 32 * 6), ymm6);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                // Use "{" and "}" to avoid the gcc warnings
                if (N >= 8) {
                    _mm256_storeu_si256((__m256i *)(dest + 32 * 7), ymm7);
                    std::atomic_signal_fence(std::memory_order_release);
                }

                dest += kSingleLoopBytes;
            }

            avx_forward_move_N_tailing<T, kSrcIsAligned, kDestIsNotAligned, _N - 1>(dest, src, end);
        } else {
#if defined(JSTD_IS_ICC)
#pragma code_align(64)
#endif
            while (src < limit) {
                __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
                if (N >= 0) {
                    ymm0 = _mm256_loadu_si256((const __m256i *)(src + 32 * 0));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 2) {
                    ymm1 = _mm256_loadu_si256((const __m256i *)(src + 32 * 1));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 3) {
                    ymm2 = _mm256_loadu_si256((const __m256i *)(src + 32 * 2));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 4) {
                    ymm3 = _mm256_loadu_si256((const __m256i *)(src + 32 * 3));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 5) {
                    ymm4 = _mm256_loadu_si256((const __m256i *)(src + 32 * 4));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 6) {
                    ymm5 = _mm256_loadu_si256((const __m256i *)(src + 32 * 5));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 7) {
                    ymm6 = _mm256_loadu_si256((const __m256i *)(src + 32 * 6));
                    std::atomic_signal_fence(std::memory_order_release);
                }
                // Use "{" and "}" to avoid the gcc warnings
                if (N >= 8) {
                    ymm7 = _mm256_loadu_si256((const __m256i *)(src + 32 * 7));
                    std::atomic_signal_fence(std::memory_order_release);
                }

                if (kUsePrefetchHint) {
                    // Here, N would be best a multiple of 2.
                    _mm_prefetch((const char *)(src + kPrefetchOffset + 64 * 0), kPrefetchHintLevel);
                    if (N >= 3)
                    _mm_prefetch((const char *)(src + kPrefetchOffset + 64 * 1), kPrefetchHintLevel);
                    if (N >= 5)
                    _mm_prefetch((const char *)(src + kPrefetchOffset + 64 * 2), kPrefetchHintLevel);
                    if (N >= 7)
                    _mm_prefetch((const char *)(src + kPrefetchOffset + 64 * 3), kPrefetchHintLevel);
                }

                src += kSingleLoopBytes;

                if (N >= 0) {
                    _mm256_storeu_si256((__m256i *)(dest + 32 * 0), ymm0);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 2) {
                    _mm256_storeu_si256((__m256i *)(dest + 32 * 1), ymm1);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 3) {
                    _mm256_storeu_si256((__m256i *)(dest + 32 * 2), ymm2);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 4) {
                    _mm256_storeu_si256((__m256i *)(dest + 32 * 3), ymm3);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 5) {
                    _mm256_storeu_si256((__m256i *)(dest + 32 * 4), ymm4);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 6) {
                    _mm256_storeu_si256((__m256i *)(dest + 32 * 5), ymm5);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                if (N >= 7) {
                    _mm256_storeu_si256((__m256i *)(dest + 32 * 6), ymm6);
                    std::atomic_signal_fence(std::memory_order_release);
                }
                // Use "{" and "}" to avoid the gcc warnings
                if (N >= 8) {
                    _mm256_storeu_si256((__m256i *)(dest + 32 * 7), ymm7);
                    std::atomic_signal_fence(std::memory_order_release);
                }

                dest += kSingleLoopBytes;
            }

            avx_forward_move_N_tailing<T, kSrcIsNotAligned, kDestIsNotAligned, _N - 1>(dest, src, end);
        }
    }
}

#if defined(JSTD_IS_PURE_GCC)
#pragma GCC pop_options
#endif

} // namespace jstd

int main()
{
    std::vector<int> array;
    array.resize(10000000);
    for (size_t i = 0; i < array.size(); i++) {
        array[i] = i;
    }

    //auto iter = jstd::left_rotate(array.begin(), array.begin() + 32, array.end());

    //jstd::avx_forward_move_N_load_aligned<int, 8>(&array[0], &array[0] + 32, &array[0] + array.size());
    jstd::avx_forward_move_N_store_aligned<int, 8>(&array[0], &array[0] + 32, &array[0] + array.size());

    for (size_t i = 0; i < 100; i++) {
        printf("%d, ", array[i]);
    }
    return 0;
}
