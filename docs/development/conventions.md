# Conventions

## Styling

We follow the standards of the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html), but have adapted some of the conventions to better fit our needs. Differences and more detailed explanations of some rules are highlighted below.

### Casting

If an implicit cast for a constant value integral is valid, do not use a redundant explicit cast.

```c++
template <size_t N>
class BigInt {};

// DO
BigInt<1> bigint;

// DON'T DO
// This will be casted redundantly as BigInt<size_t{size_t{1}}>
BigInt<size_t{1}> bigint;

// DO
size_t GetSize() { return 1; }

// Note: Continue to use explicit casts for cases that require it logically
size_t size = size_t{1} << bit;
```

[Correlating Google C++ casting rule here](https://google.github.io/styleguide/cppguide.html#Casting)

### Comments

#### Code-specific Names

Enclose code-specific names in the comments with vertical bars "|". This includes names of variables, functions, files, objects, and more.

```c++
// DON'T DO:
// "foo" is 10
// `foo` is 10

// DO:
// |foo| is 10
size_t foo = 10;
```

[Correlating Google C++ commenting rule here](https://google.github.io/styleguide/cppguide.html#Function_Comments)

#### Greek Letters

Greek letters are often used in explanations of mathematical processes in our comments. We use the "Small" version of Greek unicode letters to standardize and assist searching capabilites.

For example:

- Use θ (U+03B8 -- "Greek Small Letter Theta")
- Do not use Θ (U+0398 -- "Greek Capital Letter Theta")

### Build Targets

Since styling Bazel files is not specified in the Google C++ style guide, here are the rules we abide by:

1. Order all `tachyon_cc_unittest` targets after all `tachyon_cc_library` targets.
2. Order targets alphabetically within each section.
3. There should be only one unittest target per Bazel file. Name the unittest target name following the pattern `{directory-name}_unittests`. An example would be `base_unittests`.

Refer to [`tachyon/base/BUILD.bazel`](tachyon/base/BUILD.bazel) as an example.

## Logging

We use `VLOG` messages in our code to show us the progress of the current running processes. 3 levels are currently employed, though more can be created as needed. Here is what our 3 levels signify:

1. `VLOG(1)` FYI - FOR YOUR INFORMATION

    Provides insight to basic information of the current process. Examples include:
  
      - Proving schemes/elliptic curves/parameters currently in use
      - Time between steps of processes
      - Start and end times of processes

2. `VLOG(2)` FOR DEBUGGING

    Exposes inner progress of the current process for easier debugging. Examples include:

      - Values of variables such as theta, beta, gamma, challenge, etc

3. `VLOG(3)` FOR TRACING

    Details outputs designed to help tracing. Examples include:

      - Output of the proof
      - Output of the verifying key serialization

## Commits

### Commit Type Classification

Deciding the commit type can be difficult if the change appears to fall within the definitions of multiple different types. Here are further clarifications on how to determine the proper commit type given previous cases we have come across.

- Any modifications made to files associated with the build process, such as `BUILD.bazel`, are categorized as a **build** commit. This classification includes changes, typo corrections, and formatting.
- Commits that add or modify tests are classified as **test** commits, while commits that remove tests are considered **refac** commits.

For more information on our Commit Message Format, see [here](https://github.com/kroma-network/.github/blob/main/CONTRIBUTING.md#commit-message-format).

## GPU

### Naming

- If your gpu-based programs can be run independent to GPU vendors, please name them `Gpu`, not `Cuda` or `ROCm`.

### File Suffixes

- If a file contains kernels, append suffixes `.cu.xx` even though it can be run on `ROCm`.
- If a file contains device-only codes, append suffixes `.cu.xx` even though it can be run on `ROCm`.
- If a file contains host and device codes, then suffixes are not appended.
  See files under `tachyon/devices/gpu` for examples.
