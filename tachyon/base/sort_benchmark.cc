#include "benchmark/benchmark.h"

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/random.h"
#include "tachyon/base/sort.h"

namespace tachyon::math {

enum class SortMethod {
  kPdq,
  kPowersort,
  kStdStableSort,
  kStdSort,
};

std::vector<uint64_t> GetData(size_t size) {
  static std::map<size_t, std::vector<uint64_t>>* s_data_map = nullptr;
  if (s_data_map == nullptr) {
    s_data_map = new std::map<size_t, std::vector<uint64_t>>();
  }
  std::vector<uint64_t>& data = (*s_data_map)[size];
  if (data.empty()) {
    data = base::CreateVector(size, [](size_t i) {
      return base::Uniform(base::Range<uint64_t>::All());
    });
  }
  return data;
}

std::vector<uint64_t> GetPartiallySortedData(size_t size) {
  static std::map<size_t, std::vector<uint64_t>>* s_data_map = nullptr;
  if (s_data_map == nullptr) {
    s_data_map = new std::map<size_t, std::vector<uint64_t>>();
  }
  std::vector<uint64_t>& data = (*s_data_map)[size];
  if (data.empty()) {
    data = base::CreateVector(size, [](size_t i) { return uint64_t{i}; });
    size_t shuffle_count = size / 8;
    for (size_t i = 0; i < shuffle_count; ++i) {
      size_t idx = base::Uniform(base::Range<size_t>::Until(shuffle_count));
      size_t idx2 = base::Uniform(base::Range<size_t>::Until(shuffle_count));
      std::swap(data[idx], data[idx2]);
    }
  }
  return data;
}

template <SortMethod kSortMethod>
void BM_SortRandomData(benchmark::State& state) {
  std::vector<uint64_t> data = GetData(state.range(0));
  std::vector<uint64_t> data2 = data;
  for (auto _ : state) {
    if constexpr (kSortMethod == SortMethod::kPdq) {
      base::UnstableSort(data2.begin(), data2.end());
    } else if constexpr (kSortMethod == SortMethod::kPowersort) {
      base::StableSort(data2.begin(), data2.end());
    } else if constexpr (kSortMethod == SortMethod::kStdStableSort) {
      std::stable_sort(data2.begin(), data2.end());
    } else if constexpr (kSortMethod == SortMethod::kStdSort) {
      std::sort(data2.begin(), data2.end());
    }
    data2 = data;
  }
  benchmark::DoNotOptimize(data);
  benchmark::DoNotOptimize(data2);
}

template <SortMethod kSortMethod>
void BM_SortPartiallySortedData(benchmark::State& state) {
  std::vector<uint64_t> data = GetPartiallySortedData(state.range(0));
  std::vector<uint64_t> data2 = data;
  for (auto _ : state) {
    if constexpr (kSortMethod == SortMethod::kPdq) {
      base::UnstableSort(data2.begin(), data2.end());
    } else if constexpr (kSortMethod == SortMethod::kPowersort) {
      base::StableSort(data2.begin(), data2.end());
    } else if constexpr (kSortMethod == SortMethod::kStdStableSort) {
      std::stable_sort(data2.begin(), data2.end());
    } else if constexpr (kSortMethod == SortMethod::kStdSort) {
      std::sort(data2.begin(), data2.end());
    }
    data2 = data;
  }
  benchmark::DoNotOptimize(data);
  benchmark::DoNotOptimize(data2);
}

BENCHMARK_TEMPLATE(BM_SortRandomData, SortMethod::kPdq)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 20);
BENCHMARK_TEMPLATE(BM_SortRandomData, SortMethod::kStdSort)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 20);
BENCHMARK_TEMPLATE(BM_SortRandomData, SortMethod::kPowersort)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 20);
BENCHMARK_TEMPLATE(BM_SortRandomData, SortMethod::kStdStableSort)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 20);

BENCHMARK_TEMPLATE(BM_SortPartiallySortedData, SortMethod::kPdq)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 20);
BENCHMARK_TEMPLATE(BM_SortPartiallySortedData, SortMethod::kStdSort)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 20);
BENCHMARK_TEMPLATE(BM_SortPartiallySortedData, SortMethod::kPowersort)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 20);
BENCHMARK_TEMPLATE(BM_SortPartiallySortedData, SortMethod::kStdStableSort)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 20);

}  // namespace tachyon::math

// clang-format off
// Executing tests from //tachyon/base:sort_benchmark
// -----------------------------------------------------------------------------
// 2024-08-05T09:26:02+00:00
// Running /home/chokobole/.cache/bazel/_bazel_chokobole/234690e3562329d13f7f07caac03dae4/execroot/kroma_network_tachyon/bazel-out/k8-opt/bin/tachyon/base/sort_benchmark.runfiles/kroma_network_tachyon/tachyon/base/sort_benchmark
// Run on (32 X 5499.96 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 2048 KiB (x16)
//   L3 Unified 36864 KiB (x1)
// Load Average: 2.58, 2.64, 2.46
// ---------------------------------------------------------------------------------------------------------
// Benchmark                                                               Time             CPU   Iterations
// ---------------------------------------------------------------------------------------------------------
// BM_SortRandomData<SortMethod::kPdq>/32                               64.5 ns         64.5 ns     10927050
// BM_SortRandomData<SortMethod::kPdq>/64                                173 ns          173 ns      4082994
// BM_SortRandomData<SortMethod::kPdq>/128                               458 ns          458 ns      1526526
// BM_SortRandomData<SortMethod::kPdq>/256                               917 ns          917 ns       767867
// BM_SortRandomData<SortMethod::kPdq>/512                              1994 ns         1994 ns       349757
// BM_SortRandomData<SortMethod::kPdq>/1024                             4694 ns         4694 ns       148456
// BM_SortRandomData<SortMethod::kPdq>/2048                             9822 ns         9822 ns        71312
// BM_SortRandomData<SortMethod::kPdq>/4096                            29664 ns        29660 ns        23724
// BM_SortRandomData<SortMethod::kPdq>/8192                            87432 ns        87430 ns         8014
// BM_SortRandomData<SortMethod::kPdq>/16384                          206323 ns       206317 ns         3389
// BM_SortRandomData<SortMethod::kPdq>/32768                          449141 ns       449055 ns         1560
// BM_SortRandomData<SortMethod::kPdq>/65536                          951382 ns       951308 ns          736
// BM_SortRandomData<SortMethod::kPdq>/131072                        1971393 ns      1971368 ns          355
// BM_SortRandomData<SortMethod::kPdq>/262144                        4258559 ns      4258174 ns          166
// BM_SortRandomData<SortMethod::kPdq>/524288                        8783038 ns      8781649 ns           71
// BM_SortRandomData<SortMethod::kPdq>/1048576                      18092094 ns     18091328 ns           39
// BM_SortRandomData<SortMethod::kStdSort>/32                           65.7 ns         65.7 ns     10562672
// BM_SortRandomData<SortMethod::kStdSort>/64                            135 ns          135 ns      5318656
// BM_SortRandomData<SortMethod::kStdSort>/128                           369 ns          369 ns      1917457
// BM_SortRandomData<SortMethod::kStdSort>/256                           780 ns          780 ns       906629
// BM_SortRandomData<SortMethod::kStdSort>/512                          1720 ns         1720 ns       402973
// BM_SortRandomData<SortMethod::kStdSort>/1024                         3937 ns         3937 ns       178163
// BM_SortRandomData<SortMethod::kStdSort>/2048                        22613 ns        22612 ns        30836
// BM_SortRandomData<SortMethod::kStdSort>/4096                        84293 ns        84293 ns         8256
// BM_SortRandomData<SortMethod::kStdSort>/8192                       212055 ns       212048 ns         3296
// BM_SortRandomData<SortMethod::kStdSort>/16384                      463883 ns       463879 ns         1505
// BM_SortRandomData<SortMethod::kStdSort>/32768                     1011653 ns      1011613 ns          692
// BM_SortRandomData<SortMethod::kStdSort>/65536                     2219845 ns      2219834 ns          315
// BM_SortRandomData<SortMethod::kStdSort>/131072                    4715038 ns      4714815 ns          149
// BM_SortRandomData<SortMethod::kStdSort>/262144                   10003182 ns     10002959 ns           70
// BM_SortRandomData<SortMethod::kStdSort>/524288                   20982786 ns     20982067 ns           33
// BM_SortRandomData<SortMethod::kStdSort>/1048576                  43885082 ns     43884019 ns           16
// BM_SortRandomData<SortMethod::kPowersort>/32                         84.1 ns         84.1 ns      8342074
// BM_SortRandomData<SortMethod::kPowersort>/64                          190 ns          190 ns      3655272
// BM_SortRandomData<SortMethod::kPowersort>/128                         446 ns          446 ns      1580095
// BM_SortRandomData<SortMethod::kPowersort>/256                         984 ns          984 ns       711206
// BM_SortRandomData<SortMethod::kPowersort>/512                        2340 ns         2340 ns       302783
// BM_SortRandomData<SortMethod::kPowersort>/1024                       5165 ns         5164 ns       130697
// BM_SortRandomData<SortMethod::kPowersort>/2048                      33551 ns        33551 ns        21216
// BM_SortRandomData<SortMethod::kPowersort>/4096                     100123 ns       100120 ns         6908
// BM_SortRandomData<SortMethod::kPowersort>/8192                     246248 ns       246236 ns         2893
// BM_SortRandomData<SortMethod::kPowersort>/16384                    544985 ns       544968 ns         1285
// BM_SortRandomData<SortMethod::kPowersort>/32768                   1185154 ns      1185143 ns          590
// BM_SortRandomData<SortMethod::kPowersort>/65536                   2544378 ns      2544265 ns          275
// BM_SortRandomData<SortMethod::kPowersort>/131072                  5465731 ns      5465444 ns          128
// BM_SortRandomData<SortMethod::kPowersort>/262144                 11856806 ns     11856582 ns           59
// BM_SortRandomData<SortMethod::kPowersort>/524288                 25558533 ns     25557610 ns           27
// BM_SortRandomData<SortMethod::kPowersort>/1048576                53946952 ns     53946318 ns           12
// BM_SortRandomData<SortMethod::kStdStableSort>/32                      106 ns          106 ns      6572790
// BM_SortRandomData<SortMethod::kStdStableSort>/64                      233 ns          233 ns      2997692
// BM_SortRandomData<SortMethod::kStdStableSort>/128                     497 ns          497 ns      1416719
// BM_SortRandomData<SortMethod::kStdStableSort>/256                    1133 ns         1133 ns       614965
// BM_SortRandomData<SortMethod::kStdStableSort>/512                    2474 ns         2473 ns       281798
// BM_SortRandomData<SortMethod::kStdStableSort>/1024                   5541 ns         5541 ns       126485
// BM_SortRandomData<SortMethod::kStdStableSort>/2048                  39425 ns        39424 ns        17762
// BM_SortRandomData<SortMethod::kStdStableSort>/4096                  99459 ns        99456 ns         7006
// BM_SortRandomData<SortMethod::kStdStableSort>/8192                 238091 ns       238082 ns         2953
// BM_SortRandomData<SortMethod::kStdStableSort>/16384                539281 ns       539263 ns         1300
// BM_SortRandomData<SortMethod::kStdStableSort>/32768               1168490 ns      1168483 ns          597
// BM_SortRandomData<SortMethod::kStdStableSort>/65536               2514056 ns      2513922 ns          279
// BM_SortRandomData<SortMethod::kStdStableSort>/131072              5325962 ns      5325863 ns          131
// BM_SortRandomData<SortMethod::kStdStableSort>/262144             11590090 ns     11589633 ns           61
// BM_SortRandomData<SortMethod::kStdStableSort>/524288             24997070 ns     24993431 ns           29
// BM_SortRandomData<SortMethod::kStdStableSort>/1048576            52364991 ns     52364081 ns           13
// BM_SortPartiallySortedData<SortMethod::kPdq>/32                      26.1 ns         26.1 ns     27818789
// BM_SortPartiallySortedData<SortMethod::kPdq>/64                      67.9 ns         67.9 ns     10076970
// BM_SortPartiallySortedData<SortMethod::kPdq>/128                      159 ns          159 ns      4460125
// BM_SortPartiallySortedData<SortMethod::kPdq>/256                      326 ns          326 ns      2155077
// BM_SortPartiallySortedData<SortMethod::kPdq>/512                      612 ns          612 ns      1158772
// BM_SortPartiallySortedData<SortMethod::kPdq>/1024                    1205 ns         1205 ns       587540
// BM_SortPartiallySortedData<SortMethod::kPdq>/2048                    2252 ns         2252 ns       309958
// BM_SortPartiallySortedData<SortMethod::kPdq>/4096                    4450 ns         4450 ns       157626
// BM_SortPartiallySortedData<SortMethod::kPdq>/8192                    9237 ns         9237 ns        75456
// BM_SortPartiallySortedData<SortMethod::kPdq>/16384                  19177 ns        19176 ns        36507
// BM_SortPartiallySortedData<SortMethod::kPdq>/32768                  48304 ns        48302 ns        14462
// BM_SortPartiallySortedData<SortMethod::kPdq>/65536                 124313 ns       124310 ns         5590
// BM_SortPartiallySortedData<SortMethod::kPdq>/131072                282868 ns       282862 ns         2471
// BM_SortPartiallySortedData<SortMethod::kPdq>/262144                748262 ns       748224 ns          930
// BM_SortPartiallySortedData<SortMethod::kPdq>/524288               1587168 ns      1587159 ns          441
// BM_SortPartiallySortedData<SortMethod::kPdq>/1048576              3434923 ns      3434856 ns          204
// BM_SortPartiallySortedData<SortMethod::kStdSort>/32                  37.8 ns         37.8 ns     17865125
// BM_SortPartiallySortedData<SortMethod::kStdSort>/64                  81.7 ns         81.7 ns      8572684
// BM_SortPartiallySortedData<SortMethod::kStdSort>/128                  201 ns          201 ns      3502366
// BM_SortPartiallySortedData<SortMethod::kStdSort>/256                  459 ns          459 ns      1513415
// BM_SortPartiallySortedData<SortMethod::kStdSort>/512                 1124 ns         1124 ns       622606
// BM_SortPartiallySortedData<SortMethod::kStdSort>/1024                2566 ns         2566 ns       274741
// BM_SortPartiallySortedData<SortMethod::kStdSort>/2048                5570 ns         5570 ns       125864
// BM_SortPartiallySortedData<SortMethod::kStdSort>/4096               12436 ns        12435 ns        56418
// BM_SortPartiallySortedData<SortMethod::kStdSort>/8192               26709 ns        26708 ns        26237
// BM_SortPartiallySortedData<SortMethod::kStdSort>/16384              69598 ns        69597 ns        10023
// BM_SortPartiallySortedData<SortMethod::kStdSort>/32768             187689 ns       187657 ns         3598
// BM_SortPartiallySortedData<SortMethod::kStdSort>/65536             422855 ns       422759 ns         1658
// BM_SortPartiallySortedData<SortMethod::kStdSort>/131072            926691 ns       926666 ns          755
// BM_SortPartiallySortedData<SortMethod::kStdSort>/262144           2141686 ns      2141672 ns          327
// BM_SortPartiallySortedData<SortMethod::kStdSort>/524288           4528703 ns      4528676 ns          153
// BM_SortPartiallySortedData<SortMethod::kStdSort>/1048576          9690864 ns      9690796 ns           72
// BM_SortPartiallySortedData<SortMethod::kPowersort>/32                47.6 ns         47.6 ns     14566847
// BM_SortPartiallySortedData<SortMethod::kPowersort>/64                69.7 ns         69.7 ns     10073432
// BM_SortPartiallySortedData<SortMethod::kPowersort>/128                119 ns          119 ns      5879687
// BM_SortPartiallySortedData<SortMethod::kPowersort>/256                234 ns          234 ns      2997032
// BM_SortPartiallySortedData<SortMethod::kPowersort>/512                472 ns          472 ns      1489784
// BM_SortPartiallySortedData<SortMethod::kPowersort>/1024               959 ns          958 ns       729569
// BM_SortPartiallySortedData<SortMethod::kPowersort>/2048              1963 ns         1963 ns       358399
// BM_SortPartiallySortedData<SortMethod::kPowersort>/4096              4909 ns         4908 ns       142478
// BM_SortPartiallySortedData<SortMethod::kPowersort>/8192             10384 ns        10381 ns        67528
// BM_SortPartiallySortedData<SortMethod::kPowersort>/16384            43864 ns        43863 ns        15994
// BM_SortPartiallySortedData<SortMethod::kPowersort>/32768           118543 ns       118541 ns         5922
// BM_SortPartiallySortedData<SortMethod::kPowersort>/65536           282220 ns       282215 ns         2485
// BM_SortPartiallySortedData<SortMethod::kPowersort>/131072          639465 ns       639359 ns         1093
// BM_SortPartiallySortedData<SortMethod::kPowersort>/262144         1601955 ns      1601811 ns          435
// BM_SortPartiallySortedData<SortMethod::kPowersort>/524288         3426178 ns      3425631 ns          206
// BM_SortPartiallySortedData<SortMethod::kPowersort>/1048576        7438610 ns      7438426 ns           91
// BM_SortPartiallySortedData<SortMethod::kStdStableSort>/32            70.3 ns         70.3 ns     10067148
// BM_SortPartiallySortedData<SortMethod::kStdStableSort>/64             131 ns          131 ns      5360170
// BM_SortPartiallySortedData<SortMethod::kStdStableSort>/128            266 ns          266 ns      2622105
// BM_SortPartiallySortedData<SortMethod::kStdStableSort>/256            631 ns          631 ns      1074449
// BM_SortPartiallySortedData<SortMethod::kStdStableSort>/512           1243 ns         1243 ns       566539
// BM_SortPartiallySortedData<SortMethod::kStdStableSort>/1024          2805 ns         2805 ns       249929
// BM_SortPartiallySortedData<SortMethod::kStdStableSort>/2048          5767 ns         5767 ns       121282
// BM_SortPartiallySortedData<SortMethod::kStdStableSort>/4096         12814 ns        12813 ns        55051
// BM_SortPartiallySortedData<SortMethod::kStdStableSort>/8192         34371 ns        34370 ns        20305
// BM_SortPartiallySortedData<SortMethod::kStdStableSort>/16384        95992 ns        95990 ns         7274
// BM_SortPartiallySortedData<SortMethod::kStdStableSort>/32768       201706 ns       201701 ns         3472
// BM_SortPartiallySortedData<SortMethod::kStdStableSort>/65536       453366 ns       453346 ns         1543
// BM_SortPartiallySortedData<SortMethod::kStdStableSort>/131072      977696 ns       977652 ns          715
// BM_SortPartiallySortedData<SortMethod::kStdStableSort>/262144     2291897 ns      2291858 ns          305
// BM_SortPartiallySortedData<SortMethod::kStdStableSort>/524288     5089614 ns      5089385 ns          137
// BM_SortPartiallySortedData<SortMethod::kStdStableSort>/1048576   11416225 ns     11415938 ns           60
// clang-format on
