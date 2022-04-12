
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

    __m128i partOfInsert = _mm256_extractf128_si256(target, index >> 3);
    partOfInsert = _mm_insert_epi16(partOfInsert, value, index % 8);
    __m256i result = _mm256_insertf128_si256(target, partOfInsert, index >> 3);
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
    short * pcontent = (short *)&content[0];
    result = insert0(target, pcontent[0]);
    result = insert4(target, pcontent[1]);
    result = insert8(target, pcontent[2]);
    result = insert12(target, pcontent[3]);

    __m256i result2 = _mm256_set1_epi16(pcontent[64]);

    char buffer[33];
    _mm256_store_si256((__m256i *)&buffer[0], result);
    buffer[32] = '\0';
    printf("Buffer: %s", buffer);
    _mm256_store_si256((__m256i *)&buffer[0], result2);
    buffer[32] = '\0';
    printf("Buffer: %s", buffer);
    return 0;
}
