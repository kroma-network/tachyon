#include "tachyon/math/finite_fields/goldilocks_prime/goldilocks_prime_field.h"

namespace tachyon::math {

// static
void GoldilocksConfig::Init() { GoldilocksGmp::Init(); }

}  // namespace tachyon::math
