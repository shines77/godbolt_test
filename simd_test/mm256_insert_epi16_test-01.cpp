
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <immintrin.h>

template <int index>
static inline
__m256i mm256_insert_epi16(__m256i target, int value)
{
    static_assert((index >= 0 && index < 16), "AVX::mm256_insert_epi16(): index must be [0-15]");
    __m256i result;
#if defined(__AVX2__)
    if (index == 0 && index < 8) {
#if 0
        __m128i target128 = _mm256_castsi256_si128(target);
        __m128i low_mixed128 = _mm_insert_epi16(target128, value, (index < 8) ? index : 0);
        result = _mm256_inserti128_si256(target, low_mixed128, 0);
#else            
        // There maybe is a bug because the value of the high 128 bits maybe lost.
        __m128i target128 = _mm256_castsi256_si128(target);
        __m128i result128 = _mm_insert_epi16(target128, value, (index < 8) ? index : 0);
        result = _mm256_castsi128_si256(result128);
#endif
    }
    else if (index >= 8 && index < 16) {
        __m128i high128 = _mm256_extracti128_si256(target, 1);
        __m128i mixed128 = _mm_insert_epi16(high128, value, (index >= 8) ? (index - 8) : 0);
        result = _mm256_inserti128_si256(target, mixed128, 1);
    }
    else {
        assert(false);
    }
#elif defined(__AVX__)
    __m128i partOf128 = _mm256_extracti128_si256(target, (index >> 3));
    __m128i mixed128 = _mm_insert_epi16(partOf128, value, (index % 8);
    result = _mm256_inserti128_si256(target, mixed128, (index >> 3));
#endif
    return result;
}

// test functions
__m256i insert0(__m256i v, int value) {
    return mm256_insert_epi16<0>(v, value);
}

__m256i insert4(__m256i v, int value) {
    return mm256_insert_epi16<4>(v, value);
}

__m256i insert8(__m256i v, int value) {
    return mm256_insert_epi16<8>(v, value);
}

__m256i insert12(__m256i v, int value) {
    return mm256_insert_epi16<12>(v, value);
}

int main(int argc, char * argv[])
{
    char * filename = nullptr;
    if (argc > 1) {
        filename = argv[1];
    }
    char content[256] = { 0 };
    FILE * file = fopen(filename, "wb+");
    fseek(file, 0, SEEK_SET);
    fread(content, sizeof(content) - 1, 255, file);
    __m256i target = _mm256_setzero_si256();
    __m256i result;
    int rnd_num;
    rnd_num = (content[0] + rand()) % 16;
    result = insert0(target, rnd_num);
    rnd_num = (content[1] + rand()) % 16;
    result = insert4(target, rnd_num);
    rnd_num = (content[2] + rand()) % 16;
    result = insert8(target, rnd_num);
    rnd_num = (content[3] + rand()) % 16;
    result = insert12(target, rnd_num);
    __m256i result2 = _mm256_set1_epi16(content[128]);
    char buffer[33];
    _mm256_store_si256((__m256i *)&buffer[0], result);
    buffer[32] = '\0';
    printf("Buffer: %s", buffer);
    _mm256_store_si256((__m256i *)&buffer[0], result2);
    buffer[32] = '\0';
    printf("Buffer: %s", buffer);
    return 0;
}
