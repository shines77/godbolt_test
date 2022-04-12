#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <inttypes.h>
#include <time.h>
#include <assert.h>
#include <immintrin.h>

#include <cstdint>
#include <cstddef>
#include <cstdbool>
#include <string>
#include <cstring>
#include <vector>
#include <algorithm>
#include <type_traits>

#include "stddef.h"
#include "CPUWarmUp.h"
#include "StopWatch.h"

extern void print_marcos();

static const size_t kDataLength = 10000000;

enum TestId {
    MUL_U64x64_X64,
    MUL_U64x64_I386,
    MUL_U64x64_ARM_V6
};

namespace jstd {

struct _uint128_t {
    typedef uint64_t    integral_t;
    typedef uint64_t    high_t;
    typedef uint64_t    low_t;

    typedef uint64_t    unsigned_t;
    typedef int64_t     signed_t;

    static const uint64_t kSignMask64   = 0x8000000000000000ull;
    static const uint64_t kBodyMask64   = 0x7FFFFFFFFFFFFFFFull;
    static const uint64_t kFullMask64   = 0xFFFFFFFFFFFFFFFFull;
    static const uint64_t kHighMask     = 0x7FFFFFFFFFFFFFFFull;
    static const uint64_t kLowMask      = 0xFFFFFFFFFFFFFFFFull;

    static const uint32_t kSignMask32   = 0x80000000ul;
    static const uint32_t kBodyMask32   = 0x7FFFFFFFul;
    static const uint32_t kFullMask32   = 0xFFFFFFFFul;
    static const uint64_t kFullMask32_64 = 0x00000000FFFFFFFFull;

    static const integral_t kZero64     = (integral_t)0ull;
    static const uint32_t   kZero32     = (uint32_t)0ull;
    static const high_t     kHighZero   = (high_t)0ull;
    static const low_t      kLowZero    = (low_t)0ull;  

    low_t  low;
    high_t high;

    _uint128_t() noexcept
        : low((low_t)0ull), high((high_t)0ull) {}

    _uint128_t(uint64_t _high, uint64_t _low) noexcept
        : low((low_t)_low), high((high_t)_high) {}

    //
    // p (128) = a (64) * b (64)
    //
    // From: https://stackoverflow.com/questions/25095741/how-can-i-multiply-64-bit-operands-and-get-128-bit-result-portably
    //
    static inline
    _uint128_t mul_u64x64(integral_t multiplicand, integral_t multiplier) {
        /*
         * GCC and Clang usually provide __uint128_t on 64-bit targets,
         * although Clang also defines it on WASM despite having to use
         * builtins for most purposes - including multiplication.
         */
#if defined(__SIZEOF_INT128__) && !defined(__wasm__)
        _uint128_t product128;
        __uint128_t product = (__uint128_t)multiplicand * multiplier;
        product128.low  = (low_t)(product & kFullMask64);
        product128.high = (high_t)(product >> 64);
        return product128;
#elif defined(_MSC_VER) && (defined(_M_IX64) || defined(_M_AMD64))
        /* Use the _umul128 intrinsic on MSVC x64 to hint for mulq. */
        _uint128_t product128;
        product128.low = _umul128(multiplicand, multiplier, &product128.high);
        return product128;
#elif defined(__ARM__) || defined(__ARM64__)
        /*
         * Fast yet simple grade school multiply that avoids
         * 64-bit carries with the properties of multiplying by 11
         * and takes advantage of UMAAL on ARMv6 to only need 4
         * calculations.
         */
        /*******************************************************************

           multiplicand (64) = low0, high0
           multiplier (64)   = low1, high1

           multiplicand (64) * multiplier (64) =

           |           |             |            |           |
           |           |             |      high0 * high1     |  product_03
           |           |       low0  * high1      |           |  product_02
           |           |       high0 * low1       |           |  product_01
           |      low0 * low1        |            |           |  product_00
           |           |             |            |           |
           0          32            64           96          128

        *******************************************************************/
        uint32_t low0  = (multiplicand & 0xFFFFFFFF);
        uint32_t high0 = (multiplicand >> 32);
        uint32_t low1  = (multiplier & 0xFFFFFFFF);
        uint32_t high1 = (multiplier >> 32);

        /* First calculate all of the cross products. */
        uint64_t product_00 = (uint64_t)low0  * low1;
        uint64_t product_01 = (uint64_t)high0 * low1;
        uint64_t product_02 = (uint64_t)low0  * high1;
        uint64_t product_03 = (uint64_t)high0 * high1;

        /* Now add the products together. These will never overflow. */
        uint64_t middle = product_02 + (uint32_t)(product_00 >> 32) + (uint32_t)(product_01 & 0xFFFFFFFFul);
        uint64_t low64  = (uint32_t)(product_00 & 0xFFFFFFFFul) | (middle << 32);
        uint64_t high64 = product_03 + (uint32_t)(product_01 >> 32) + (uint32_t)(middle >> 32);
        return _uint128_t((high_t)high64, (low_t)low64);
#else // __i386__ or other
        /*******************************************************************

           multiplicand (64) = low0, high0
           multiplier (64)   = low1, high1

           multiplicand (64) * multiplier (64) =

           |           |             |            |           |
           |           |             |      high0 * high1     |  product_03
           |           |       low0  * high1      |           |  product_02
           |           |       high0 * low1       |           |  product_01
           |      low0 * low1        |            |           |  product_00
           |           |             |            |           |
           0          32            64           96          128

        *******************************************************************/
        uint32_t low0  = (multiplicand & 0xFFFFFFFF);
        uint32_t high0 = (multiplicand >> 32);
        uint32_t low1  = (multiplier & 0xFFFFFFFF);
        uint32_t high1 = (multiplier >> 32);

        /* First calculate all of the cross products. */
        uint64_t product_00 = (uint64_t)low0  * low1;
        uint64_t product_01 = (uint64_t)high0 * low1;
        uint64_t product_02 = (uint64_t)low0  * high1;
        uint64_t product_03 = (uint64_t)high0 * high1;

        /* Now add the products together. These will never overflow. */
        uint64_t middle = product_01 + product_02 + (uint32_t)(product_00 >> 32);
        uint64_t low64  = (uint32_t)(product_00 & 0xFFFFFFFFul) | (middle << 32);
        uint64_t high64 = product_03 + (uint32_t)(middle >> 32);
        return _uint128_t((high_t)high64, (low_t)low64);
#endif // __i386__
    }

    static inline
    _uint128_t mul_u64x64_i386(integral_t multiplicand, integral_t multiplier) {
        /*******************************************************************

           multiplicand (64) = low0, high0
           multiplier (64)   = low1, high1

           multiplicand (64) * multiplier (64) =

           |           |             |            |           |
           |           |             |      high0 * high1     |  product_03
           |           |       low0  * high1      |           |  product_02
           |           |       high0 * low1       |           |  product_01
           |      low0 * low1        |            |           |  product_00
           |           |             |            |           |
           0          32            64           96          128

        *******************************************************************/
        uint32_t low0  = (multiplicand & 0xFFFFFFFF);
        uint32_t high0 = (multiplicand >> 32);
        uint32_t low1  = (multiplier & 0xFFFFFFFF);
        uint32_t high1 = (multiplier >> 32);

        /* First calculate all of the cross products. */
        uint64_t product_00 = (uint64_t)low0  * low1;
        uint64_t product_01 = (uint64_t)high0 * low1;
        uint64_t product_02 = (uint64_t)low0  * high1;
        uint64_t product_03 = (uint64_t)high0 * high1;

        /* Now add the products together. These will never overflow. */
        uint64_t middle = product_01 + product_02 + (uint32_t)(product_00 >> 32);
        uint64_t low64  = (uint32_t)(product_00 & 0xFFFFFFFFul) | (middle << 32);
        uint64_t high64 = product_03 + (uint32_t)(middle >> 32);
        return _uint128_t((high_t)high64, (low_t)low64);
    }

    static inline
    _uint128_t mul_u64x64_armv6(integral_t multiplicand, integral_t multiplier) {
        /*
         * Fast yet simple grade school multiply that avoids
         * 64-bit carries with the properties of multiplying by 11
         * and takes advantage of UMAAL on ARMv6 to only need 4
         * calculations.
         */
        /*******************************************************************

           multiplicand (64) = low0, high0
           multiplier (64)   = low1, high1

           multiplicand (64) * multiplier (64) =

           |           |             |            |           |
           |           |             |      high0 * high1     |  product_03
           |           |       low0  * high1      |           |  product_02
           |           |       high0 * low1       |           |  product_01
           |      low0 * low1        |            |           |  product_00
           |           |             |            |           |
           0          32            64           96          128

        *******************************************************************/
        uint32_t low0  = (multiplicand & 0xFFFFFFFF);
        uint32_t high0 = (multiplicand >> 32);
        uint32_t low1  = (multiplier & 0xFFFFFFFF);
        uint32_t high1 = (multiplier >> 32);

        /* First calculate all of the cross products. */
        uint64_t product_00 = (uint64_t)low0  * low1;
        uint64_t product_01 = (uint64_t)high0 * low1;
        uint64_t product_02 = (uint64_t)low0  * high1;
        uint64_t product_03 = (uint64_t)high0 * high1;

        /* Now add the products together. These will never overflow. */
        uint64_t middle = product_02 + (uint32_t)(product_00 >> 32) + (uint32_t)(product_01 & 0xFFFFFFFFul);
        uint64_t low64  = (uint32_t)(product_00 & 0xFFFFFFFFul) | (middle << 32);
        uint64_t high64 = product_03 + (uint32_t)(product_01 >> 32) + (uint32_t)(middle >> 32);
        return _uint128_t((high_t)high64, (low_t)low64);
    }
};

} // namespace jstd

static inline
uint32_t next_random_u32()
{
#if (RAND_MAX == 0x7FFF)
    uint32_t rnd32 = (((uint32_t)rand() & 0x03) << 30) |
                      ((uint32_t)rand() << 15) |
                       (uint32_t)rand();
#else
    uint32_t rnd32 = ((uint32_t)rand() << 16) | (uint32_t)rand();
#endif
    return rnd32;
}

static inline
uint64_t next_random_u64()
{
#if (RAND_MAX == 0x7FFF)
    uint64_t rnd64 = (((uint64_t)rand() & 0x0F) << 60) |
                      ((uint64_t)rand() << 45) |
                      ((uint64_t)rand() << 30) |
                      ((uint64_t)rand() << 15) |
                       (uint64_t)rand();
#else
    uint64_t rnd64 = ((uint64_t)rand() << 32) | (uint64_t)rand();
#endif
    return rnd64;
}

void print_mul_u64x64_result(uint64_t a, uint64_t b, const jstd::_uint128_t & product)
{
    printf("a = %" PRIu64 ", b = %" PRIu64 "\n", a, b);
    printf("product.high = 0x%08X%08X, product.low = 0x%08X%08X\n\n",
           (uint32_t)(product.high >> 32), (uint32_t)(product.high & 0xFFFFFFFFul),
           (uint32_t)(product.low >> 32), (uint32_t)(product.low & 0xFFFFFFFFul));
}

void test_const_value()
{
    uint64_t a = 0xFFFFFFFFFFFFFFFFull;
    uint64_t b = 0xFFFFFFFFFFFFFFFFull;
    jstd::_uint128_t product128 = jstd::_uint128_t::mul_u64x64(a, b);
    print_mul_u64x64_result(a, b, product128);
}

void test_random_value()
{
    uint64_t a = next_random_u64();
    uint64_t b = next_random_u64();
    jstd::_uint128_t product128 = jstd::_uint128_t::mul_u64x64(a, b);
    print_mul_u64x64_result(a, b, product128);
}

void test_random_value_i386(uint64_t a, uint64_t b)
{
    jstd::_uint128_t product128 = jstd::_uint128_t::mul_u64x64_i386(a, b);
    print_mul_u64x64_result(a, b, product128);
}

void test_random_value_armv6(uint64_t a, uint64_t b)
{
    jstd::_uint128_t product128 = jstd::_uint128_t::mul_u64x64_armv6(a, b);
    print_mul_u64x64_result(a, b, product128);
}

template <int test_id>
void run_mul_u64x64_benchmark(const std::vector<uint64_t> & test_data, std::string name)
{
    test::StopWatch sw;
    uint64_t check_sum = 0;

    sw.start();
    if (test_id == MUL_U64x64_X64) {
        for (size_t i = 0; i < test_data.size(); i += 2) {
            jstd::_uint128_t product128 = jstd::_uint128_t::mul_u64x64(test_data[i], test_data[i + 1]);
            check_sum += product128.low + product128.high;
        }
    }
    else if (test_id == MUL_U64x64_I386) {
        for (size_t i = 0; i < test_data.size(); i += 2) {
            jstd::_uint128_t product128 = jstd::_uint128_t::mul_u64x64_i386(test_data[i], test_data[i + 1]);
            check_sum += product128.low + product128.high;
        }
    }
    else if (test_id == MUL_U64x64_ARM_V6) {
        for (size_t i = 0; i < test_data.size(); i += 2) {
            jstd::_uint128_t product128 = jstd::_uint128_t::mul_u64x64_armv6(test_data[i], test_data[i + 1]);
            check_sum += product128.low + product128.high;
        }
    }
    else {
        name = "Unknown test function";
    }
    sw.stop();

    double elapsedTime = sw.getElapsedMillisec();

    printf("===============================================================================\n\n");
    printf("function = %s()\n", name.c_str());
    printf("check_sum = %" PRIu64 ", elapsedTime: %0.2f ms\n\n", check_sum, elapsedTime);
}

void mul_u64x64_benchmark()
{
    test::CPU::WarmUp warm_up(1000);

    std::vector<uint64_t> test_data;
    test_data.resize(kDataLength * 2);
    for (size_t i = 0; i < kDataLength * 2; i++) {
        test_data[i] = next_random_u64();
    }

    run_mul_u64x64_benchmark<MUL_U64x64_X64>   (test_data, "mul_u64x64_x64");
    run_mul_u64x64_benchmark<MUL_U64x64_I386>  (test_data, "mul_u64x64_i386");
    run_mul_u64x64_benchmark<MUL_U64x64_ARM_V6>(test_data, "mul_u64x64_armv6");

    printf("===============================================================================\n");
    printf("\n");
}

int main(int argc, char * argv[])
{
    srand((unsigned)time(NULL));

    print_marcos();

    test_const_value();
    test_random_value();
    test_random_value();

    uint64_t a = next_random_u64();
    uint64_t b = next_random_u64();

    test_random_value_i386(a, b);
    test_random_value_armv6(a, b);

    mul_u64x64_benchmark();

    return 0;
}
