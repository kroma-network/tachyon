// clang-format off
#include "tachyon/base/logging.h"
#include "tachyon/math/finite_fields/fp%{degree}.h"
#include "%{base_field_hdr}"

namespace %{namespace} {

template <typename _BaseField>
class %{class}Config {
 public:
  using BaseField = _BaseField;
  using BasePrimeField = %{base_prime_field};
  using FrobeniusCoefficient = %{frobenius_coefficient};

  // TODO(chokobole): Make them constexpr.
  static BaseField kNonResidue;
  static FrobeniusCoefficient kFrobeniusCoeffs[%{frobenius_coeffs_size}];
%{if FrobeniusCoefficient2}
  static FrobeniusCoefficient kFrobeniusCoeffs2[%{frobenius_coeffs_size}];
%{endif FrobeniusCoefficient2}
%{if FrobeniusCoefficient3}
  static FrobeniusCoefficient kFrobeniusCoeffs3[%{frobenius_coeffs_size}];
%{endif FrobeniusCoefficient3}

  constexpr static bool kNonResidueIsMinusOne = %{non_residue_is_minus_one};
  constexpr static uint64_t kDegreeOverBaseField = %{degree_over_base_field};
  constexpr static uint64_t kDegreeOverBasePrimeField = %{degree_over_base_prime_field};

  static BaseField MulByNonResidue(const BaseField& v) {
%{mul_by_non_residue_code}
  }

  static void Init() {
    BaseField::Init();
%{init_code}
    VLOG(1) << "%{namespace}::%{class} initialized";
  }
};

template <typename BaseField>
BaseField %{class}Config<BaseField>::kNonResidue;
template <typename BaseField>
typename %{class}Config<BaseField>::FrobeniusCoefficient %{class}Config<BaseField>::kFrobeniusCoeffs[%{frobenius_coeffs_size}];
%{if FrobeniusCoefficient2}
template <typename BaseField>
typename %{class}Config<BaseField>::FrobeniusCoefficient %{class}Config<BaseField>::kFrobeniusCoeffs2[%{frobenius_coeffs_size}];
%{endif FrobeniusCoefficient2}
%{if FrobeniusCoefficient3}
template <typename BaseField>
typename %{class}Config<BaseField>::FrobeniusCoefficient %{class}Config<BaseField>::kFrobeniusCoeffs3[%{frobenius_coeffs_size}];
%{endif FrobeniusCoefficient3}
using %{class} = Fp%{degree}<%{class}Config<%{base_field}>>;

}  // namespace %{namespace}
// clang-format on
