#include <string>

#include "rust/cxx.h"

namespace tachyon {

struct CppG1Affine;
struct CppG1Jacobian;
struct CppFr;

rust::Vec<uint64_t> get_test_nums(rust::Slice<const rust::String> argv);

void arkworks_benchmark(rust::Slice<const rust::u64> test_nums,
                        rust::Slice<const CppG1Affine> bases,
                        rust::Slice<const CppFr> scalars,
                        rust::Slice<const CppG1Jacobian> results_arkworks,
                        rust::Slice<const rust::f64> durations_arkworks);

}  // namespace tachyon
