# Convention

## C++ Style Guide

### Casting

It's allowed to avoid explicit casting when an integral value is used as a constant and
can be implicitly cast. See [casting](https://google.github.io/styleguide/cppguide.html#Casting).

```c++
// Using an integral value as a constant without explicit casting is allowed
template <size_t N>
class BigInt {};

BigInt<1> bigint;         // This is okay
BigInt<size_t{1}> bigint; // This is not allowed

// When using a constant integral value, casting is not necessary
size_t GetSize() { return 1; } // This is okay

// However, when performing operations like bit-shifting, it's recommended
// to follow the google c++ style guide.
size_t size = size_t{1} << bit; // Recommended to cast as per the style guide
```

## GPU

### Naming

- If your gpu codes can be run independent to GPU vendors, please use a word `Gpu` not `Cuda` or `ROCm`.

### File Suffixes

- If a file contains kernels, there will be suffixes `.cu.xx` despite it can be run on `ROCm`.
- If a file contains device only codes, there will be suffixes `.cu.xx` despite it can be run on `ROCm`.
- If a file contains host and device codes, then suffixes are not appended.
  For examples, see files under `tachyon/devices/gpu`.

## Commit

### Commit Type Classification

Deciding the commit type can be difficult if the change appears to fall within the definitions of multiple different types. Here are further clarifications on how to determine the proper commit type given previous cases we have come across.

- Any modifications made to files associated with the build process, such as `BUILD.bazel`, are categorized as a **build** commit. This classification includes changes, typo corrections, and formatting as well.
- **test** commits are for cases where only tests are added or modified. Commits that remove tests are classified as **refac**.

For more information on our Commit Message Format, see [here](https://github.com/kroma-network/.github/blob/main/CONTRIBUTING.md#commit-message-format).
