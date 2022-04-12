
#include <immintrin.h>
#include <stdint.h>

template <int index>
static inline
__m256i mm256_insert_epi64(__m256i target, int64_t value)
{
    __m128i partOfInsert = _mm256_extractf128_si256(target, index >> 1);
    partOfInsert = _mm_insert_epi64(partOfInsert, value, index % 2);
    __m256i result = _mm256_insertf128_si256 (target, partOfInsert, index >> 1);
    return result;
}

// test functions
__m256i insert0(__m256i v, int64_t newval) {
    return mm256_insert_epi64<0>(v, newval);
}

__m256i insert1(__m256i v, int64_t newval) {
    return mm256_insert_epi64<1>(v, newval);
}

__m256i insert2(__m256i v, int64_t newval) {
    return mm256_insert_epi64<2>(v, newval);
}

__m256i insert3(__m256i v, int64_t newval) {
    return mm256_insert_epi64<3>(v, newval);
}
