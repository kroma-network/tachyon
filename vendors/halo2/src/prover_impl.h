#ifndef VENDORS_HALO2_SRC_PROVER_IMPL_H_
#define VENDORS_HALO2_SRC_PROVER_IMPL_H_

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <utility>

#include "rust/cxx.h"

#include "tachyon/base/functional/callback.h"
#include "tachyon/zk/plonk/halo2/prover.h"

namespace tachyon::halo2_api {

template <typename PCS>
class ProverImpl {
 public:
  explicit ProverImpl(base::OnceCallback<zk::halo2::Prover<PCS>()> callback)
      : prover_(std::move(callback).Run()) {}

  const zk::halo2::Prover<PCS>& prover() const { return prover_; }

  size_t K() const { return prover_.pcs().K(); }

  size_t N() const { return prover_.pcs().N(); }

 private:
  zk::halo2::Prover<PCS> prover_;
};

}  // namespace tachyon::halo2_api

#endif  // VENDORS_HALO2_SRC_PROVER_IMPL_H_
