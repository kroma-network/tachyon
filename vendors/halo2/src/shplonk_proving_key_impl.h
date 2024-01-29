#ifndef VENDORS_HALO2_SRC_SHPLONK_PROVING_KEY_IMPL_H_
#define VENDORS_HALO2_SRC_SHPLONK_PROVING_KEY_IMPL_H_

#include <utility>
#include <vector>

#include "rust/cxx.h"

#include "tachyon/base/buffer/buffer.h"
#include "tachyon/zk/base/commitments/shplonk_extension.h"
#include "tachyon/zk/plonk/halo2/pinned_verifying_key.h"
#include "tachyon/zk/plonk/keys/proving_key.h"
#include "vendors/halo2/include/proving_key_impl_forward.h"
#include "vendors/halo2/src/buffer_reader.h"

namespace tachyon::halo2_api {

template <typename Curve, size_t MaxDegree, size_t MaxExtendedDegree>
class ProvingKeyImpl<
    zk::SHPlonkExtension<Curve, MaxDegree, MaxExtendedDegree,
                         math::AffinePoint<typename Curve::G1Curve>>> {
 public:
  using Commitment = math::AffinePoint<typename Curve::G1Curve>;
  using PCS =
      zk::SHPlonkExtension<Curve, MaxDegree, MaxExtendedDegree, Commitment>;
  using F = typename PCS::Field;
  using Evals = typename PCS::Evals;
  using Poly = typename PCS::Poly;

  explicit ProvingKeyImpl(rust::Slice<const uint8_t> bytes) {
    base::Buffer buffer(const_cast<uint8_t*>(bytes.data()), bytes.size());
    ReadProvingKey(buffer);
  }

  const zk::ProvingKey<PCS>& key() const { return key_; }

  const zk::ConstraintSystem<F>& GetConstraintSystem() const {
    return key_.verifying_key().constraint_system();
  }

  const std::vector<zk::Phase>& GetAdviceColumnPhases() const {
    return GetConstraintSystem().advice_column_phases();
  }

  size_t ComputeBlindingFactors() const {
    return GetConstraintSystem().ComputeBlindingFactors();
  }

  const std::vector<zk::Phase>& GetChallengePhases() const {
    return GetConstraintSystem().challenge_phases();
  }

  const std::vector<zk::FixedColumnKey> GetConstants() const {
    return GetConstraintSystem().constants();
  }

  size_t GetNumAdviceColumns() const {
    return GetConstraintSystem().num_advice_columns();
  }

  size_t GetNumChallenges() const {
    return GetConstraintSystem().num_challenges();
  }

  size_t GetNumInstanceColumns() const {
    return GetConstraintSystem().num_instance_columns();
  }

  std::vector<zk::Phase> GetPhases() const {
    return GetConstraintSystem().GetPhases();
  }

  const F& GetTranscriptRepr(const zk::Entity<PCS>& entity) {
    key_.verifying_key_.SetTranscriptRepresentative(&entity);
    return key_.verifying_key_.transcript_repr_;
  }

 private:
  void ReadProvingKey(base::Buffer& buffer) {
    ReadVerifyingKey(buffer, key_.verifying_key_);
    ReadBuffer(buffer, key_.l_first_);
    ReadBuffer(buffer, key_.l_last_);
    ReadBuffer(buffer, key_.l_active_row_);
    ReadBuffer(buffer, key_.fixed_columns_);
    ReadBuffer(buffer, key_.fixed_polys_);
    ReadBuffer(buffer, key_.permutation_proving_key_);
    key_.vanishing_argument_ = zk::VanishingArgument<F>::Create(
        key_.verifying_key_.constraint_system());
    CHECK(buffer.Done());
  }

  static void ReadVerifyingKey(base::Buffer& buffer,
                               zk::VerifyingKey<PCS>& vkey) {
    // NOTE(chokobole): For k
    ReadU32AsSizeT(buffer);
    ReadBuffer(buffer, vkey.fixed_commitments_);
    ReadConstraintSystem(buffer, vkey.constraint_system_);
    size_t num_commitments =
        vkey.constraint_system_.permutation().columns().size();
    std::vector<Commitment> commitments;
    commitments.resize(num_commitments);
    for (size_t i = 0; i < num_commitments; ++i) {
      ReadBuffer(buffer, commitments[i]);
    }
    vkey.permutation_verifying_key_ =
        zk::PermutationVerifyingKey<Commitment>(std::move(commitments));
  }

  static void ReadConstraintSystem(base::Buffer& buffer,
                                   zk::ConstraintSystem<F>& cs) {
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
    cs.num_advice_queries_ = base::CreateVector(
        ReadU32AsSizeT(buffer),
        [&buffer]() { return BufferReader<uint32_t>::Read(buffer); });
    ReadBuffer(buffer, cs.instance_queries_);
    ReadBuffer(buffer, cs.fixed_queries_);
    ReadBuffer(buffer, cs.permutation_);
    ReadBuffer(buffer, cs.lookups_);
    ReadBuffer(buffer, cs.constants_);
  }

  zk::ProvingKey<PCS> key_;
};

}  // namespace tachyon::halo2_api

#endif  // VENDORS_HALO2_SRC_SHPLONK_PROVING_KEY_IMPL_H_
