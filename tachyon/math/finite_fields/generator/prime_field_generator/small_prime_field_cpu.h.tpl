// clang-format off
#include "%{config_header_path}"

%{if kUseMontgomery}
#include "tachyon/math/finite_fields/small_prime_field_mont.h"
%{endif kUseMontgomery}
%{if !kUseMontgomery}
#include "tachyon/math/finite_fields/small_prime_field.h"
%{endif !kUseMontgomery}

namespace %{namespace} {

using %{class} = PrimeField<%{class}Config>;

}  // namespace %{namespace}
// clang-format on
