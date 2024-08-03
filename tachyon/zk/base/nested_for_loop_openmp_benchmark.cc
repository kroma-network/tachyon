#include "benchmark/benchmark.h"

#include "tachyon/base/parallelize.h"
#include "tachyon/math/elliptic_curves/bn/bn254/fr.h"

namespace tachyon::zk {

template <typename F>
void BM_NestedForLoopParallelCols(benchmark::State& state) {
  size_t cols = state.range(0);
  size_t rows = state.range(1);
  std::vector<std::vector<F>> table =
      base::CreateVectorParallel(cols, [rows]() {
        return base::CreateVector(rows, []() { return F::Random(); });
      });
  for (auto _ : state) {
    OMP_PARALLEL_FOR(size_t i = 0; i < cols; ++i) {
      for (size_t j = 0; j < rows; ++j) {
        table[i][j].DoubleInPlace();
      }
    }
  }
  benchmark::DoNotOptimize(table);
}

template <typename F>
void BM_NestedForLoopParallelRows(benchmark::State& state) {
  size_t cols = state.range(0);
  size_t rows = state.range(1);
  std::vector<std::vector<F>> table =
      base::CreateVectorParallel(cols, [rows]() {
        return base::CreateVector(rows, []() { return F::Random(); });
      });
  for (auto _ : state) {
    for (size_t i = 0; i < cols; ++i) {
      OMP_PARALLEL_FOR(size_t j = 0; j < rows; ++j) {
        table[i][j].DoubleInPlace();
      }
    }
  }
  benchmark::DoNotOptimize(table);
}

template <typename F>
void BM_NestedForLoopParallelCollapse(benchmark::State& state) {
  size_t cols = state.range(0);
  size_t rows = state.range(1);
  std::vector<std::vector<F>> table =
      base::CreateVectorParallel(cols, [rows]() {
        return base::CreateVector(rows, []() { return F::Random(); });
      });
  for (auto _ : state) {
    OMP_PARALLEL_NESTED_FOR(size_t i = 0; i < cols; ++i) {
      for (size_t j = 0; j < rows; ++j) {
        table[i][j].DoubleInPlace();
      }
    }
  }
  benchmark::DoNotOptimize(table);
}

BENCHMARK_TEMPLATE(BM_NestedForLoopParallelCols, math::bn254::Fr)
    ->ArgsProduct({benchmark::CreateDenseRange(500, 1000, /*step=*/100),
                   benchmark::CreateRange(1 << 10, 1 << 15, /*multi=*/2)});

BENCHMARK_TEMPLATE(BM_NestedForLoopParallelRows, math::bn254::Fr)
    ->ArgsProduct({benchmark::CreateDenseRange(500, 1000, /*step=*/100),
                   benchmark::CreateRange(1 << 10, 1 << 15, /*multi=*/2)});

BENCHMARK_TEMPLATE(BM_NestedForLoopParallelCollapse, math::bn254::Fr)
    ->ArgsProduct({benchmark::CreateDenseRange(500, 1000, /*step=*/100),
                   benchmark::CreateRange(1 << 10, 1 << 15, /*multi=*/2)});

}  // namespace tachyon::zk

// clang-format off
// Executing tests from //tachyon/zk/base:nested_for_loop_openmp_benchmark
// -----------------------------------------------------------------------------
// 2024-02-14T01:39:58+00:00
// Running /home/ryan/.cache/bazel/_bazel_ryan/d6800124b8b6155cc6ab653ae18dfdd6/execroot/kroma_network_tachyon/bazel-out/k8-opt/bin/tachyon/zk/base/nested_for_loop_openmp_benchmark.runfiles/kroma_network_tachyon/tachyon/zk/base/nested_for_loop_openmp_benchmark
// Run on (32 X 5500 MHz CPU s)
// CPU Caches:
//   L1 Data 48 KiB (x16)
//   L1 Instruction 32 KiB (x16)
//   L2 Unified 2048 KiB (x16)
//   L3 Unified 36864 KiB (x1)
// Load Average: 0.26, 0.17, 0.19
// -------------------------------------------------------------------------------------------------------
// Benchmark                                                             Time             CPU   Iterations
// -------------------------------------------------------------------------------------------------------
// BM_NestedForLoopParallelCols<math::bn254::Fr>/500/1024          8801567 ns      8040507 ns           87
// BM_NestedForLoopParallelCols<math::bn254::Fr>/600/1024          8925674 ns      7830494 ns           85
// BM_NestedForLoopParallelCols<math::bn254::Fr>/700/1024          9003939 ns      7779225 ns          112
// BM_NestedForLoopParallelCols<math::bn254::Fr>/800/1024          9030484 ns      7531339 ns           91
// BM_NestedForLoopParallelCols<math::bn254::Fr>/900/1024          8995042 ns      7910739 ns          121
// BM_NestedForLoopParallelCols<math::bn254::Fr>/1000/1024         9050964 ns      7898973 ns           89
// BM_NestedForLoopParallelCols<math::bn254::Fr>/500/2048          9021839 ns      8085116 ns           95
// BM_NestedForLoopParallelCols<math::bn254::Fr>/600/2048          9036271 ns      8287476 ns           84
// BM_NestedForLoopParallelCols<math::bn254::Fr>/700/2048          9254811 ns      8349779 ns          108
// BM_NestedForLoopParallelCols<math::bn254::Fr>/800/2048          9513442 ns      8219528 ns           89
// BM_NestedForLoopParallelCols<math::bn254::Fr>/900/2048          9563439 ns      7934262 ns          100
// BM_NestedForLoopParallelCols<math::bn254::Fr>/1000/2048         9932232 ns      8465107 ns           80
// BM_NestedForLoopParallelCols<math::bn254::Fr>/500/4096          9961692 ns      8186292 ns           90
// BM_NestedForLoopParallelCols<math::bn254::Fr>/600/4096         10435311 ns      8437591 ns           89
// BM_NestedForLoopParallelCols<math::bn254::Fr>/700/4096         10956990 ns     10013659 ns           75
// BM_NestedForLoopParallelCols<math::bn254::Fr>/800/4096         11579415 ns     10750777 ns           68
// BM_NestedForLoopParallelCols<math::bn254::Fr>/900/4096         12267116 ns      9536249 ns           77
// BM_NestedForLoopParallelCols<math::bn254::Fr>/1000/4096        12359321 ns     11485318 ns           60
// BM_NestedForLoopParallelCols<math::bn254::Fr>/500/8192         12697713 ns     10886044 ns           60
// BM_NestedForLoopParallelCols<math::bn254::Fr>/600/8192         13833057 ns     11446369 ns           62
// BM_NestedForLoopParallelCols<math::bn254::Fr>/700/8192         14833471 ns     14832698 ns           47
// BM_NestedForLoopParallelCols<math::bn254::Fr>/800/8192         16094893 ns     12794323 ns           47
// BM_NestedForLoopParallelCols<math::bn254::Fr>/900/8192         17802661 ns     15956079 ns           48
// BM_NestedForLoopParallelCols<math::bn254::Fr>/1000/8192        18653991 ns     16865193 ns           59
// BM_NestedForLoopParallelCols<math::bn254::Fr>/500/16384        18098826 ns     16406897 ns           47
// BM_NestedForLoopParallelCols<math::bn254::Fr>/600/16384        20269987 ns     16955481 ns           39
// BM_NestedForLoopParallelCols<math::bn254::Fr>/700/16384        21743041 ns     17145256 ns           40
// BM_NestedForLoopParallelCols<math::bn254::Fr>/800/16384        23719447 ns     21584281 ns           35
// BM_NestedForLoopParallelCols<math::bn254::Fr>/900/16384        25161989 ns     21385216 ns           33
// BM_NestedForLoopParallelCols<math::bn254::Fr>/1000/16384       27030126 ns     23671931 ns           30
// BM_NestedForLoopParallelCols<math::bn254::Fr>/500/32768        27190628 ns     23602275 ns           29
// BM_NestedForLoopParallelCols<math::bn254::Fr>/600/32768        30077304 ns     24809468 ns           28
// BM_NestedForLoopParallelCols<math::bn254::Fr>/700/32768        33541336 ns     29339242 ns           25
// BM_NestedForLoopParallelCols<math::bn254::Fr>/800/32768        36750392 ns     32140843 ns           22
// BM_NestedForLoopParallelCols<math::bn254::Fr>/900/32768        41558349 ns     36509732 ns           20
// BM_NestedForLoopParallelCols<math::bn254::Fr>/1000/32768       45558915 ns     37758209 ns           17
// BM_NestedForLoopParallelRows<math::bn254::Fr>/500/1024       4336841822 ns   4285953805 ns            1
// BM_NestedForLoopParallelRows<math::bn254::Fr>/600/1024       5208263159 ns   5058216868 ns            1
// BM_NestedForLoopParallelRows<math::bn254::Fr>/700/1024       6044492960 ns   5915146990 ns            1
// BM_NestedForLoopParallelRows<math::bn254::Fr>/800/1024       6922183037 ns   5931739080 ns            1
// BM_NestedForLoopParallelRows<math::bn254::Fr>/900/1024       7746754408 ns   6681311598 ns            1
// BM_NestedForLoopParallelRows<math::bn254::Fr>/1000/1024      8651807547 ns   8186728777 ns            1
// BM_NestedForLoopParallelRows<math::bn254::Fr>/500/2048       4287798643 ns   3932394790 ns            1
// BM_NestedForLoopParallelRows<math::bn254::Fr>/600/2048       5186981201 ns   4775116743 ns            1
// BM_NestedForLoopParallelRows<math::bn254::Fr>/700/2048       6069154024 ns   5310053507 ns            1
// BM_NestedForLoopParallelRows<math::bn254::Fr>/800/2048       6981294632 ns   6459602603 ns            1
// BM_NestedForLoopParallelRows<math::bn254::Fr>/900/2048       2702628613 ns   2357686183 ns            1
// BM_NestedForLoopParallelRows<math::bn254::Fr>/1000/2048      1880954981 ns   1789799983 ns            1
// BM_NestedForLoopParallelRows<math::bn254::Fr>/500/4096          4988417 ns      4717985 ns          264
// BM_NestedForLoopParallelRows<math::bn254::Fr>/600/4096         28701242 ns     26118750 ns          159
// BM_NestedForLoopParallelRows<math::bn254::Fr>/700/4096          8012837 ns      7494953 ns          102
// BM_NestedForLoopParallelRows<math::bn254::Fr>/800/4096          8716901 ns      8209507 ns          133
// BM_NestedForLoopParallelRows<math::bn254::Fr>/900/4096         11234994 ns     10559360 ns          107
// BM_NestedForLoopParallelRows<math::bn254::Fr>/1000/4096         7201372 ns      7201110 ns          101
// BM_NestedForLoopParallelRows<math::bn254::Fr>/500/8192          9088564 ns      8945928 ns          124
// BM_NestedForLoopParallelRows<math::bn254::Fr>/600/8192         12592125 ns     12379448 ns           95
// BM_NestedForLoopParallelRows<math::bn254::Fr>/700/8192         21225240 ns     20438229 ns           54
// BM_NestedForLoopParallelRows<math::bn254::Fr>/800/8192         16714320 ns     16178267 ns           50
// BM_NestedForLoopParallelRows<math::bn254::Fr>/900/8192         42304723 ns     40716275 ns           61
// BM_NestedForLoopParallelRows<math::bn254::Fr>/1000/8192        22066152 ns     21179481 ns           54
// BM_NestedForLoopParallelRows<math::bn254::Fr>/500/16384        34603290 ns     33077561 ns           57
// BM_NestedForLoopParallelRows<math::bn254::Fr>/600/16384      1448661089 ns   1259494978 ns            1
// BM_NestedForLoopParallelRows<math::bn254::Fr>/700/16384        50462682 ns     47181769 ns           29
// BM_NestedForLoopParallelRows<math::bn254::Fr>/800/16384        23063511 ns     22644394 ns           37
// BM_NestedForLoopParallelRows<math::bn254::Fr>/900/16384        51485399 ns     48692191 ns           36
// BM_NestedForLoopParallelRows<math::bn254::Fr>/1000/16384      176147251 ns    161074240 ns           17
// BM_NestedForLoopParallelRows<math::bn254::Fr>/500/32768        78185296 ns     73173143 ns           10
// ^BBM_NestedForLoopParallelRows<math::bn254::Fr>/600/32768       149321532 ns    139500894 ns           10
// BM_NestedForLoopParallelRows<math::bn254::Fr>/700/32768       291554535 ns    250161912 ns           17
// BM_NestedForLoopParallelRows<math::bn254::Fr>/800/32768       130893564 ns    120231909 ns           10
// BM_NestedForLoopParallelRows<math::bn254::Fr>/900/32768       115641832 ns    109627786 ns           11
// BM_NestedForLoopParallelRows<math::bn254::Fr>/1000/32768       91647482 ns     86392369 ns           15
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/500/1024      3230082 ns      2961001 ns         4752
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/600/1024      4188581 ns      3587240 ns         1000
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/700/1024      2469267 ns      2197792 ns         3084
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/800/1024       665211 ns       611305 ns          820
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/900/1024       258700 ns       257955 ns         2382
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/1000/1024      583019 ns       557772 ns         1843
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/500/2048      1686582 ns      1535978 ns         2364
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/600/2048      4188626 ns      3440178 ns         1000
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/700/2048      1336820 ns      1198356 ns         1587
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/800/2048      2190726 ns      2052386 ns         1000
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/900/2048      1762969 ns      1530832 ns         1138
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/1000/2048     7961306 ns      6703722 ns          520
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/500/4096     11053157 ns      9109342 ns           80
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/600/4096      9912193 ns      8139388 ns           72
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/700/4096     11803536 ns      9460880 ns           68
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/800/4096     11462906 ns      9122242 ns           67
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/900/4096      7112131 ns      6236221 ns          100
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/1000/4096     8327859 ns      7121217 ns          149
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/500/8192      8801355 ns      7689890 ns          149
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/600/8192     10712903 ns     10024969 ns          121
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/700/8192      9197602 ns      8394859 ns          102
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/800/8192     12644033 ns     11226135 ns           86
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/900/8192     12511659 ns     11929592 ns           77
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/1000/8192    11700689 ns     10882559 ns           73
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/500/16384    11866962 ns     11114133 ns           68
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/600/16384    13764891 ns     13641851 ns           58
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/700/16384    16557857 ns     16087721 ns           51
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/800/16384    20248668 ns     19096909 ns           43
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/900/16384    22054337 ns     21400207 ns           32
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/1000/16384   25220033 ns     20827619 ns           35
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/500/32768    21271888 ns     21097723 ns           34
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/600/32768    28138599 ns     25294890 ns           25
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/700/32768    34027233 ns     30336595 ns           25
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/800/32768    40258440 ns     33080362 ns           22
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/900/32768    38469553 ns     33483133 ns           19
// BM_NestedForLoopParallelCollapse<math::bn254::Fr>/1000/32768   47346022 ns     36566534 ns           18
// clang-format on
