# Godbolt.org compiler explorer

## 1. Source code

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

int main(int argc, char * argv[])
{
    printf("Hello world.\n\n");
    return 0;
}
```

## 2. Link

Test URL: [HelloWord.cpp](https://www.godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGe1wAyeAyYAHI%2BAEaYxBJmpAAOqAqETgwe3r56icmOAkEh4SxRMVxxdpgOqUIETMQE6T5%2BXLaY9rkM1bUE%2BWGR0bG2NXUNmc0KQ93BvUX9pQCUtqhexMjsHCYaAILmAMzByN5YANQmO27j%2BKgAdAin2BvbZnsMB17Hp%2BcE6LR4ETd3D12%2B0OmBOZwuwQI/x29y2QJeILBbiYCiUdWhsO2W0hRxYTGCEBxtWAyFIR2QCFqRwAVEdiQA3EwAVisTIAInNAQB2KxbI78o7xYiQqgQcxmAASrVoqCOAHcSLR0FdmW4GKr1WYzJydrzNgKjsRMARlgwjhpTnqTFy2Q8OAtaJwmbw/BwtKRUJw3NZrEcFEsVqDdjxSARNPaFgBrEA7HZXACcZiZ8YAHGmAGw7Lhcrla/ScSQu8Mezi8BQgDSh8MLOCwJBoFjxOjRciUBtN%2BgxYBcUp8OgEaLliARYsRYK1ACenBDDbYggA8gxaFO3bwsHijOJV6R8EbKvTMOXt5hVBUvAPp7xIa1iz8IsRJx4sJfQ8KWJeFlQDMAFAA1PCYHK87xIwL78IIIhiOwUgyIIigqOo266HEBhGCgPqWPovzlpACyoPE7RHgAtBcpxssADBeBhFhSEcRHzjsdEAOpiLQdF4istw7GylIKHK0plq0FTtC4DDuJ4jT%2BGJPSFMUWRJCkAgjE0CQKe0Ml9CULRtFUEzKXo5SVAInR1BpMxaeMXT6WMExmXJXALP6yyrHoBBGmsPAOk6Rbbp6HCqCm6ZEemkhHCSyBHD2VxmEcEDepY1hkrghAkCcTzNEcHiNs2xBpTs2q8GGq5zAsCCYEwWAxBAUYxpIVwaDsGjpqmXIaEyOySFyTJJvmHCFqQrrun5ZYVlWxWkLWiAoKg2Wdq2EDtjlKCod2ZgaM0NC0AOxBDiO25jswxArjOM1zgQi7LsW66GMAW7uruwl4AeR7uieZ4XtwV6CDe253g%2BR1Ph5hVvh%2BfDfn%2BAFASBrohuBwiiOIMFw/BajFrozQrehCWYXeOHVR6BGpMRDHMax7G1BSZG8fxtC0IJOnOBArjWaQgRTLJsyqTkqQs9kikMHZnOGe0Jn1BJozaY9xm2ezmkGXp4sqZZpmy%2BZEiOQGLnNG5mBA15HDOgNxZ%2BQFQUheSK2RWY9VXFwsXxVYmFHMlRC5cGZJZR20R5Q5hXVjV%2BX1ZIsalMHUhdT1jp9bw75cBolaDbww22KNRVaCVvVmD5Q2lmN6cLAeO2pCAkhAA)

Text:

```text
https://www.godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGe1wAyeAyYAHI%2BAEaYxBJmpAAOqAqETgwe3r56icmOAkEh4SxRMVxxdpgOqUIETMQE6T5%2BXLaY9rkM1bUE%2BWGR0bG2NXUNmc0KQ93BvUX9pQCUtqhexMjsHCYaAILmAMzByN5YANQmO27j%2BKgAdAin2BvbZnsMB17Hp%2BcE6LR4ETd3D12%2B0OmBOZwuwQI/x29y2QJeILBbiYCiUdWhsO2W0hRxYTGCEBxtWAyFIR2QCFqRwAVEdiQA3EwAVisTIAInNAQB2KxbI78o7xYiQqgQcxmAASrVoqCOAHcSLR0FdmW4GKr1WYzJydrzNgKjsRMARlgwjhpTnqTFy2Q8OAtaJwmbw/BwtKRUJw3NZrEcFEsVqDdjxSARNPaFgBrEA7HZXACcZiZ8YAHGmAGw7Lhcrla/ScSQu8Mezi8BQgDSh8MLOCwJBoFjxOjRciUBtN%2BgxYBcUp8OgEaLliARYsRYK1ACenBDDbYggA8gxaFO3bwsHijOJV6R8EbKvTMOXt5hVBUvAPp7xIa1iz8IsRJx4sJfQ8KWJeFlQDMAFAA1PCYHK87xIwL78IIIhiOwUgyIIigqOo266HEBhGCgPqWPovzlpACyoPE7RHgAtBcpxssADBeBhFhSEcRHzjsdEAOpiLQdF4istw7GylIKHK0plq0FTtC4DDuJ4jT%2BGJPSFMUWRJCkAgjE0CQKe0Ml9CULRtFUEzKXo5SVAInR1BpMxaeMXT6WMExmXJXALP6yyrHoBBGmsPAOk6Rbbp6HCqCm6ZEemkhHCSyBHD2VxmEcEDepY1hkrghAkCcTzNEcHiNs2xBpTs2q8GGq5zAsCCYEwWAxBAUYxpIVwaDsGjpqmXIaEyOySFyTJJvmHCFqQrrun5ZYVlWxWkLWiAoKg2Wdq2EDtjlKCod2ZgaM0NC0AOxBDiO25jswxArjOM1zgQi7LsW66GMAW7uruwl4AeR7uieZ4XtwV6CDe253g%2BR1Ph5hVvh%2BfDfn%2BAFASBrohuBwiiOIMFw/BajFrozQrehCWYXeOHVR6BGpMRDHMax7G1BSZG8fxtC0IJOnOBArjWaQgRTLJsyqTkqQs9kikMHZnOGe0Jn1BJozaY9xm2ezmkGXp4sqZZpmy%2BZEiOQGLnNG5mBA15HDOgNxZ%2BQFQUheSK2RWY9VXFwsXxVYmFHMlRC5cGZJZR20R5Q5hXVjV%2BX1ZIsalMHUhdT1jp9bw75cBolaDbww22KNRVaCVvVmD5Q2lmN6cLAeO2pCAkhAA
```

## 3. Compiler

* x86-64 gcc 11.2:

    ```text
    -std=c++11 -O3 -Wall -march=native -DNDEBUG
    -std=c++11 -O3 -Wall -march=skylake -DNDEBUG
    -std=gnu++14 -O3 -Wall -march=haswell -DNDEBUG
    ```

    ```text
    '-march=' switch are: nocona core2 nehalem corei7 westmere sandybridge corei7-avx ivybridge core-avx-i haswell core-avx2 broadwell skylake skylake-avx512 cannonlake icelake-client rocketlake icelake-server cascadelake tigerlake cooperlake sapphirerapids alderlake bonnell atom silvermont slm goldmont goldmont-plus tremont knl knm x86-64 x86-64-v2 x86-64-v3 x86-64-v4 eden-x2 nano nano-1000 nano-2000 nano-3000 nano-x2 eden-x4 nano-x4 k8 k8-sse3 opteron opteron-sse3 athlon64 athlon64-sse3 athlon-fx amdfam10 barcelona bdver1 bdver2 bdver3 bdver4 znver1 znver2 znver3 btver1 btver2 native;
    ```

* x86-64 clang 12.0.1:

    ```text
    -std=c++11 -O3 -Wall -march=native -DNDEBUG
    -std=c++14 -O3 -Wall -march=haswell -DNDEBUG
    ```

* MSVC:

    ```bash
    ## No /MT /MD
    /O2 /W3 /WX- /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /GS /GL /Gy /Zi /Gm- /Gd /Oi /EHsc /nologo /Zc:inline /fp:precise
    ## /MT
    /O2 /W3 /WX- /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /GS /GL /Gy /Zi /Gm- /Gd /Oi /MT /EHsc /nologo /Zc:inline /fp:precise
    ## /MD
    /O2 /W3 /WX- /D "NDEBUG" /D "_CONSOLE" /D "_MBCS" /GS /GL /Gy /Zi /Gm- /Gd /Oi /MD /EHsc /nologo /Zc:inline /fp:precise /MACHINE:X64
    ```

## 4. Copyright

Power by [https://www.godbolt.org/](https://www.godbolt.org/)
