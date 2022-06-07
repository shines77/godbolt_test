
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <cstdbool>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <vector>
#include <atomic>
#include <algorithm>
#include <type_traits>

// hash functor
template <typename T>
struct hash {
    typedef T               argument_type;
    typedef std::size_t     result_type;

    std::size_t operator () (const argument_type & key) const {
        return 0;
    }
};

template <>
struct hash<std::int32_t> {
    typedef std::int32_t    argument_type;
    typedef std::size_t     result_type;

    std::size_t operator () (std::int32_t key) const {
        return ((std::size_t)key * 2654435761ull);
    }
};

template <>
struct hash<std::uint32_t> {
    typedef std::uint32_t   argument_type;
    typedef std::size_t     result_type;

    std::size_t operator () (std::uint32_t key) const {
        return ((std::size_t)key * 2654435761ull);
    }
};

template <>
struct hash<std::intptr_t> {
    typedef std::intptr_t   argument_type;
    typedef std::size_t     result_type;

    std::size_t operator () (std::intptr_t key) const {
        return ((std::size_t)key * 2654435761ull);
    }
};

template <>
struct hash<std::size_t> {
    typedef std::size_t     argument_type;
    typedef std::size_t     result_type;

    std::size_t operator () (std::size_t key) const {
        return (key * 2654435761ull);
    }
};

template <typename Key, typename Value, typename Hasher = std::hash<Key>>
class HashMap {
public:
    typedef Key                     key_type;
    typedef Value                   mapped_type;
    typedef std::pair<Key, Value>   value_type;
    typedef Hasher                  hasher_type;

    typedef std::size_t size_type;

    struct Entry {
        size_type   hash_code;
        key_type    key;
        mapped_type value;

        Entry() : hash_code(0) {}

        Entry(const Entry & src) :
            hash_code(src.hash_code),
            key(src.key), value(src.value) {
        }
    };

private:
    hasher_type hasher_;
    Entry entry_;

public:
    HashMap() {
    }

    HashMap(const hasher_type & hasher) {
        hasher_ = hasher;
    }

    void insert(const key_type & key, const mapped_type & value) {
        size_type hash_code = hasher_(key);
        entry_.hash_code = hash_code;
        entry_.key = key;
        entry_.value = value;
    }

    void display_entry() {
        std::cout << "hash_code: " << entry_.hash_code << '\n';
        std::cout << "key: " << entry_.key << '\n';
        std::cout << "value: " << entry_.value << '\n';
    }
};

int main()
{
    //hash<int> hasher;
    //HashMap<int, int, hash<int>> hash_map(hasher);
    HashMap<int, int> hash_map;
    hash_map.insert(33, 100);
    hash_map.display_entry();

    return 0;
}
