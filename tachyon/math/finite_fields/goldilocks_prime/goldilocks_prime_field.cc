#include "tachyon/math/finite_fields/goldilocks_prime/goldilocks_prime_field.h"

namespace tachyon {
namespace math {

// static
void GoldilocksConfig::Init() { GoldilocksGmp::Init(); }

}  // namespace math
}  // namespace tachyon
