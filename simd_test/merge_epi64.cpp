
#include <immintrin.h>
#include <stdint.h>

template<unsigned elem>
static inline
__m256i merge_epi64(__m256i v, int64_t newval)
{
    static_assert(elem <= 3, "a __m256i only has 4 qword elements");
    __m256i splat = _mm256_set1_epi64x(newval);

    constexpr unsigned dword_blendmask = 0b11 << (elem*2);  // vpblendd uses 2 bits per qword
    return  _mm256_blend_epi32(v, splat, dword_blendmask);
}

// test functions
__m256i merge0(__m256i v, int64_t newval) {
    return merge_epi64<0>(v, newval);
}

__m256i merge1(__m256i v, int64_t newval) {
    return merge_epi64<1>(v, newval);
}

__m256i merge2(__m256i v, int64_t newval) {
    return merge_epi64<2>(v, newval);
}

__m256i merge3(__m256i v, int64_t newval) {
    return merge_epi64<3>(v, newval);
}
