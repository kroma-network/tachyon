#ifndef TACHYON_C_ZK_PLONK_KEYS_PROVING_KEY_IMPL_BASE_H_
#define TACHYON_C_ZK_PLONK_KEYS_PROVING_KEY_IMPL_BASE_H_

#include <stdint.h>

#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/base/buffer/read_only_buffer.h"
#include "tachyon/base/environment.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/c/zk/plonk/keys/buffer_reader.h"
#include "tachyon/zk/plonk/halo2/pinned_verifying_key.h"
#include "tachyon/zk/plonk/keys/proving_key.h"

namespace tachyon::c::zk::plonk {

template <typename Poly, typename Evals, typename C>
class ProvingKeyImplBase
    : public tachyon::zk::plonk::ProvingKey<Poly, Evals, C> {
 public:
  using F = typename Poly::Field;

  ProvingKeyImplBase(absl::Span<const uint8_t> state, bool read_only_vk)
      : read_only_vk_(read_only_vk) {
    std::string_view pk_str;
    if (tachyon::base::Environment::Get("TACHYON_PK_LOG_PATH", &pk_str)) {
      VLOG(1) << "Save pk to: " << pk_str;
      CHECK(tachyon::base::WriteLargeFile(tachyon::base::FilePath(pk_str),
                                          state));
    }

    tachyon::base::ReadOnlyBuffer buffer(state.data(), state.size());

    ReadProvingKey(buffer);
  }

  const tachyon::zk::plonk::ConstraintSystem<F>& GetConstraintSystem() const {
    return this->verifying_key_.constraint_system_;
  }

  template <typename PCS>
  void SetTranscriptRepr(const tachyon::zk::Entity<PCS>& entity) {
    this->verifying_key_.SetTranscriptRepresentative(&entity);
  }

  template <typename PCS>
  const F& GetTranscriptRepr(const tachyon::zk::Entity<PCS>& entity) {
    return this->verifying_key_.transcript_repr_;
  }

 private:
  void ReadProvingKey(const tachyon::base::ReadOnlyBuffer& buffer) {
    ReadVerifyingKey(buffer, this->verifying_key_);
    if (read_only_vk_) return;
    ReadBuffer(buffer, this->l_first_);
    ReadBuffer(buffer, this->l_last_);
    ReadBuffer(buffer, this->l_active_row_);
    ReadBuffer(buffer, this->fixed_columns_);
    ReadBuffer(buffer, this->fixed_polys_);
    ReadBuffer(buffer, this->permutation_proving_key_);
    this->vanishing_argument_ =
        tachyon::zk::plonk::VanishingArgument<F>::Create(
            this->verifying_key_.constraint_system_);
    CHECK(buffer.Done());
  }

  static void ReadVerifyingKey(const tachyon::base::ReadOnlyBuffer& buffer,
                               tachyon::zk::plonk::VerifyingKey<F, C>& vkey) {
    // NOTE(chokobole): For k
    ReadU32AsSizeT(buffer);
    ReadBuffer(buffer, vkey.fixed_commitments_);
    ReadConstraintSystem(buffer, vkey.constraint_system_);
    size_t num_commitments =
        vkey.constraint_system_.permutation().columns().size();
    std::vector<C> commitments;
    commitments.resize(num_commitments);
    for (size_t i = 0; i < num_commitments; ++i) {
      ReadBuffer(buffer, commitments[i]);
    }
    vkey.permutation_verifying_key_ =
        tachyon::zk::plonk::PermutationVerifyingKey<C>(std::move(commitments));
  }

  static void ReadConstraintSystem(
      const tachyon::base::ReadOnlyBuffer& buffer,
      tachyon::zk::plonk::ConstraintSystem<F>& cs) {
    cs.num_fixed_columns_ = ReadU32AsSizeT(buffer);
    cs.num_advice_columns_ = ReadU32AsSizeT(buffer);
    cs.num_instance_columns_ = ReadU32AsSizeT(buffer);
    cs.num_selectors_ = ReadU32AsSizeT(buffer);
    cs.num_challenges_ = ReadU32AsSizeT(buffer);
    ReadBuffer(buffer, cs.advice_column_phases_);
    ReadBuffer(buffer, cs.challenge_phases_);
    ReadBuffer(buffer, cs.selector_map_);
    ReadBuffer(buffer, cs.gates_);
    ReadBuffer(buffer, cs.advice_queries_);
    cs.num_advice_queries_ = tachyon::base::CreateVector(
        ReadU32AsSizeT(buffer),
        [&buffer]() { return BufferReader<uint32_t>::Read(buffer); });
    ReadBuffer(buffer, cs.instance_queries_);
    ReadBuffer(buffer, cs.fixed_queries_);
    ReadBuffer(buffer, cs.permutation_);
    ReadBuffer(buffer, cs.lookups_);
    ReadBuffer(buffer, cs.constants_);
  }

 private:
  bool read_only_vk_ = false;
};

}  // namespace tachyon::c::zk::plonk

#endif  // TACHYON_C_ZK_PLONK_KEYS_PROVING_KEY_IMPL_BASE_H_
