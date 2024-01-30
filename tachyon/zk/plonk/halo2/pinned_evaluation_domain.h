// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_ZK_PLONK_HALO2_PINNED_EVALUATION_DOMAIN_H_
#define TACHYON_ZK_PLONK_HALO2_PINNED_EVALUATION_DOMAIN_H_

#include <stdint.h>

#include <string>
#include <utility>

#include "tachyon/base/strings/rust_stringifier.h"
#include "tachyon/zk/base/entities/entity.h"
#include "tachyon/zk/plonk/halo2/stringifiers/field_stringifier.h"

namespace tachyon {
namespace zk::halo2 {

template <typename F>
class PinnedEvaluationDomain {
 public:
  PinnedEvaluationDomain() = default;
  PinnedEvaluationDomain(uint32_t k, uint32_t extended_k, const F& omega)
      : k_(k), extended_k_(extended_k), omega_(omega) {}
  PinnedEvaluationDomain(uint32_t k, uint32_t extended_k, F&& omega)
      : k_(k), extended_k_(extended_k), omega_(std::move(omega)) {}
  template <typename PCS>
  explicit PinnedEvaluationDomain(const Entity<PCS>* entity)
      : k_(entity->domain()->log_size_of_group()),
        extended_k_(entity->extended_domain()->log_size_of_group()),
        omega_(entity->domain()->group_gen()) {}

  uint32_t k() const { return k_; }
  uint32_t extended_k() const { return extended_k_; }
  const F& omega() const { return omega_; }

 private:
  uint32_t k_ = 0;
  uint32_t extended_k_ = 0;
  F omega_;
};

}  // namespace zk::halo2

namespace base::internal {

template <typename F>
class RustDebugStringifier<zk::halo2::PinnedEvaluationDomain<F>> {
 public:
  static std::ostream& AppendToStream(
      std::ostream& os, RustFormatter& fmt,
      const zk::halo2::PinnedEvaluationDomain<F>& value) {
    return os << fmt.DebugStruct("PinnedEvaluationDomain")
                     .Field("k", value.k())
                     .Field("extended_k", value.extended_k())
                     .Field("omega", value.omega())
                     .Finish();
  }
};

}  // namespace base::internal
}  // namespace tachyon

#endif  // TACHYON_ZK_PLONK_HALO2_PINNED_EVALUATION_DOMAIN_H_
