#include <stdlib.h>
#include <stdio.h>
#include <cstdint>
#include <cstddef>
#include <chrono>

void cpu_warm_up(int delayMillsecs)
{
    using namespace std::chrono;
    double delayTimeLimit = (double)delayMillsecs / 1.0;
    volatile intptr_t sum = 0;

    printf("------------------------------------------\n\n");
    printf("CPU warm-up begin ...\n");

    high_resolution_clock::time_point startTime, endTime;
    duration<double, std::ratio<1, 1000>> elapsedTime;
    startTime = high_resolution_clock::now();
    do {
        for (intptr_t i = 0; i < 500; ++i) {
            sum += i;
            for (intptr_t j = 5000; j >= 0; --j) {
                sum -= j;
            }
        }
        endTime = high_resolution_clock::now();
        elapsedTime = endTime - startTime;
    } while (elapsedTime.count() < delayTimeLimit);

    printf("sum = %d, time: %0.3f ms\n", (int)sum, elapsedTime.count());
    printf("CPU warm-up end   ... \n\n");
    printf("------------------------------------------\n\n");
}

struct ListNode
{
    size_t val{0};
    ListNode *next{nullptr};
    ListNode(size_t x) : val(x) {}
    ListNode() = default;
};

int main(int argc, char * argv[])
{
    using namespace std::chrono;

    static constexpr intptr_t kMaxCount = 5000000;

    cpu_warm_up(1000);

    ListNode * head = nullptr;
    for (intptr_t i = kMaxCount - 1; i >= 0; --i) {
        ListNode * cur = new ListNode(i);
        cur->next = head;
        head = cur;
    }

    if (1) {
        auto begin = high_resolution_clock::now();
        ListNode * cur = head;
        while (cur->next != nullptr) {
            cur = cur->next;
        }
        auto end = high_resolution_clock::now();

        duration<double> elapsed = duration_cast<duration<double>>(end - begin);
        printf("cur = %p, time1: %0.3f ms\n", cur, elapsed.count() * 1000);
    }

    if (1) {
        auto begin = high_resolution_clock::now();
        ListNode * cur = head;
        while (cur->next != nullptr && cur->next->val != kMaxCount) {
            cur = cur->next;
        }
        auto end = high_resolution_clock::now();

        duration<double> elapsed = duration_cast<duration<double>>(end - begin);
        printf("cur = %p, time2: %0.3f ms\n", cur, elapsed.count() * 1000);
    }

    if (1) {
        auto begin = high_resolution_clock::now();
        ListNode * cur = head;
        while (cur->next != nullptr) {
            cur = cur->next;
        }
        auto end = high_resolution_clock::now();

        duration<double> elapsed = duration_cast<duration<double>>(end - begin);
        printf("cur = %p, time3: %0.3f ms\n", cur, elapsed.count() * 1000);
    }

    if (1) {
        auto begin = high_resolution_clock::now();
        ListNode * cur = head;
        while (cur->next != nullptr && cur->next->val != kMaxCount) {
            cur = cur->next;
        }
        auto end = high_resolution_clock::now();

        duration<double> elapsed = duration_cast<duration<double>>(end - begin);
        printf("cur = %p, time4: %0.3f ms\n", cur, elapsed.count() * 1000);
    }

    printf("\n");
    return 0;
}
