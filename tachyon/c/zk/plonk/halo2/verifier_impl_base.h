#ifndef TACHYON_C_ZK_PLONK_HALO2_VERIFIER_IMPL_BASE_H_
#define TACHYON_C_ZK_PLONK_HALO2_VERIFIER_IMPL_BASE_H_

#include <stdint.h>

#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/functional/callback.h"
#include "tachyon/zk/plonk/halo2/verifier.h"

namespace tachyon::c::zk::plonk::halo2 {

template <typename PCS>
class VerifierImplBase : public tachyon::zk::plonk::halo2::Verifier<PCS> {
 public:
  using Base = tachyon::zk::plonk::halo2::Verifier<PCS>;
  using Callback = base::OnceCallback<Base()>;
  using F = typename PCS::Field;
  using Commitment = typename PCS::Commitment;
  using Evals = typename PCS::Evals;

  VerifierImplBase(Callback callback, uint8_t transcript_type)
      : Base(std::move(callback).Run()), transcript_type_(transcript_type) {}

  uint8_t transcript_type() const { return transcript_type_; }

  [[nodiscard]] bool VerifyProof(
      const tachyon::zk::plonk::VerifyingKey<F, Commitment>& vkey,
      std::vector<std::vector<std::vector<F>>>& instance_columns_vec) {
    std::vector<std::vector<Evals>> cpp_instance_columns_vec =
        base::Map(instance_columns_vec,
                  [](std::vector<std::vector<F>>& instance_columns) {
                    return base::Map(instance_columns,
                                     [](std::vector<F>& instance_column) {
                                       return Evals(std::move(instance_column));
                                     });
                  });
    return Base::VerifyProof(vkey, cpp_instance_columns_vec);
  }

 protected:
  uint8_t transcript_type_;
};

}  // namespace tachyon::c::zk::plonk::halo2

#endif  // TACHYON_C_ZK_PLONK_HALO2_VERIFIER_IMPL_BASE_H_
