
#include <stdint.h>
#include <assert.h>
#include <immintrin.h>

template <int index>
static inline
__m256i mm256_insert_epi64(__m256i target, int64_t value)
{
    static_assert((index >= 0 && index < 4), "AVX::mm256_insert_epi64(): index must be [0-3]");
#if defined(__AVX2__)
    if (index == 0) {
        __m128i low64 = _mm_cvtsi64_si128(value);
        __m256i low256 = _mm256_castsi128_si256(low64);
        return _mm256_blend_epi32(target, low256, 0b00000011);
    }
    else if (index >= 1 && index <= 3) {
        static const int blend_mask = 0b00000011 << (index * 2);
        __m128i low64 = _mm_cvtsi64_si128(value);
        __m256i repeat256 = _mm256_broadcastq_epi64(low64);
        return _mm256_blend_epi32(target, repeat256, blend_mask);
    }
    else {
        assert(false);
        return __m256i();
    }
#elif defined(__AVX__)
    if (index >= 0 && index <= 1) {
        __m128i low128 = _mm256_castsi256_si128(target);
        __m128i low128_insert = _mm_insert_epi64(low128, value, index % 2);
        __m256i result = _mm256_insertf128_si256 (target, low128_insert, index >> 1);
        return result;
    }
    else if (index >= 2 && index <= 3) {
        __m128i high128 = _mm256_extractf128_si256(target, index >> 1);
        __m128i high128_insert = _mm_insert_epi64(high128, value, index % 2);
        __m256i result = _mm256_insertf128_si256 (target, high128_insert, index >> 1);
        return result;
    }
    else {
        assert(false);
        return __m256i();
    }
#else
    // This is original version
    __m128i partOfInsert = _mm256_extractf128_si256(target, index >> 1);
    partOfInsert = _mm_insert_epi64(partOfInsert, value, index % 2);
    __m256i result = _mm256_insertf128_si256 (target, partOfInsert, index >> 1);
    return result;
#endif
}

// test functions
__m256i insert0(__m256i v, int64_t value) {
    return mm256_insert_epi64<0>(v, value);
}

__m256i insert1(__m256i v, int64_t value) {
    return mm256_insert_epi64<1>(v, value);
}

__m256i insert2(__m256i v, int64_t value) {
    return mm256_insert_epi64<2>(v, value);
}

__m256i insert3(__m256i v, int64_t value) {
    return mm256_insert_epi64<3>(v, value);
}
