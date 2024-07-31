#ifndef TACHYON_C_ZK_PLONK_HALO2_VERIFIER_IMPL_H_
#define TACHYON_C_ZK_PLONK_HALO2_VERIFIER_IMPL_H_

#include <stdint.h>

#include <memory_resource>
#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/functional/callback.h"
#include "tachyon/zk/plonk/halo2/verifier.h"

namespace tachyon::c::zk::plonk::halo2 {

template <typename PCS, typename LS>
class VerifierImpl : public tachyon::zk::plonk::halo2::Verifier<PCS, LS> {
 public:
  using Base = tachyon::zk::plonk::halo2::Verifier<PCS, LS>;
  using Callback = tachyon::base::OnceCallback<Base()>;
  using F = typename PCS::Field;
  using Commitment = typename PCS::Commitment;
  using Evals = typename PCS::Evals;

  VerifierImpl(Callback callback, uint8_t transcript_type)
      : Base(std::move(callback).Run()), transcript_type_(transcript_type) {}

  uint8_t transcript_type() const { return transcript_type_; }

  using Base::VerifyProof;

  [[nodiscard]] bool VerifyProof(
      const tachyon::zk::plonk::VerifyingKey<F, Commitment>& vkey,
      std::vector<std::vector<std::pmr::vector<F>>>& instance_columns_vec) {
    std::vector<std::vector<Evals>> cpp_instance_columns_vec =
        tachyon::base::Map(
            instance_columns_vec,
            [](std::vector<std::pmr::vector<F>>& instance_columns) {
              return tachyon::base::Map(
                  instance_columns, [](std::pmr::vector<F>& instance_column) {
                    return Evals(std::move(instance_column));
                  });
            });
    return Base::VerifyProof(vkey, cpp_instance_columns_vec);
  }

 protected:
  uint8_t transcript_type_;
};

}  // namespace tachyon::c::zk::plonk::halo2

#endif  // TACHYON_C_ZK_PLONK_HALO2_VERIFIER_IMPL_H_
