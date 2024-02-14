// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_KZG_GWC_H_
#define TACHYON_CRYPTO_COMMITMENTS_KZG_GWC_H_

#include <array>
#include <utility>
#include <vector>

#include "gtest/gtest_prod.h"

#include "tachyon/crypto/commitments/kzg/kzg_family.h"
#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/crypto/commitments/univariate_polynomial_commitment_scheme.h"
#include "tachyon/crypto/transcripts/transcript.h"
#include "tachyon/math/elliptic_curves/pairing/pairing.h"

namespace tachyon {
namespace zk {

template <typename Curve, size_t MaxDegree, size_t MaxExtensionDegree,
          typename _Commitment = typename math::Pippenger<
              typename Curve::G1Curve::AffinePoint>::Bucket>
class GWCExtension;

}  // namespace zk

namespace crypto {

template <typename Curve, size_t MaxDegree,
          typename Commitment = typename math::Pippenger<
              typename Curve::G1Curve::AffinePoint>::Bucket>
class GWC final : public UnivariatePolynomialCommitmentScheme<
                      GWC<Curve, MaxDegree, Commitment>>,
                  public KZGFamily<typename Curve::G1Curve::AffinePoint,
                                   MaxDegree, Commitment> {
 public:
  using Base =
      UnivariatePolynomialCommitmentScheme<GWC<Curve, MaxDegree, Commitment>>;
  using G1Point = typename Curve::G1Curve::AffinePoint;
  using G2Point = typename Curve::G2Curve::AffinePoint;
  using G2Prepared = typename Curve::G2Prepared;
  using Fp12 = typename Curve::Fp12;
  using Field = typename Base::Field;
  using Poly = typename Base::Poly;
  using Point = typename Poly::Point;
  using PointDeepRef = base::DeepRef<const Point>;

  GWC() = default;
  explicit GWC(KZG<G1Point, MaxDegree, Commitment>&& kzg)
      : KZGFamily<G1Point, MaxDegree, Commitment>(std::move(kzg)) {}
  GWC(KZG<G1Point, MaxDegree, Commitment>&& kzg, G2Point&& s_g2)
      : KZGFamily<G1Point, MaxDegree, Commitment>(std::move(kzg)),
        s_g2_(std::move(s_g2)),
        g2_arr_({G2Prepared::From(G2Point::Generator()),
                 G2Prepared::From(-s_g2)}) {}

  const G2Point& s_g2() const { return s_g2_; }

  void ResizeBatchCommitments() {
    this->kzg_.ResizeBatchCommitments(
        this->batch_commitment_state_.batch_count);
  }

  std::vector<Commitment> GetBatchCommitments() {
    return this->kzg_.GetBatchCommitments(this->batch_commitment_state_);
  }

 private:
  friend class VectorCommitmentScheme<GWC<Curve, MaxDegree, Commitment>>;
  friend class UnivariatePolynomialCommitmentScheme<
      GWC<Curve, MaxDegree, Commitment>>;
  template <typename, size_t, size_t, typename>
  friend class zk::GWCExtension;
  FRIEND_TEST(GWCTest, Copyable);

  const char* Name() const { return "GWC"; }

  // UnivariatePolynomialCommitmentScheme methods
  template <typename Container>
  [[nodiscard]] bool DoCreateOpeningProof(
      const Container& poly_openings, TranscriptWriter<Commitment>* writer) {
    Field v = writer->SqueezeChallenge();
    VLOG(2) << "GWC(v): " << v.ToHexString(true);

    PolynomialOpeningGrouper<Poly> grouper;
    grouper.GroupBySinglePoint(poly_openings);

    // Group |poly_openings| to |grouped_poly_openings_vec|.
    // {x₀, [P₀, P₁, P₂]}
    // {x₁, [P₀, P₁, P₂]}
    // {x₂, [P₀, P₁, P₂, P₃]}
    // {x₃, [P₃]}
    // {x₄, [P₄]}
    const std::vector<GroupedPolynomialOpenings<Poly>>&
        grouped_poly_openings_vec = grouper.grouped_poly_openings_vec();

    this->SetBatchMode(grouped_poly_openings_vec.size());
    for (size_t i = 0; i < grouped_poly_openings_vec.size(); ++i) {
      // clang-format off
      // W₀(X) = H₀(X) / (X - x₀) = (P₀(X) - P₀(x₀)) + v(P₁(X) - P₁(x₀)) + v²(P₂(X) - P₂(x₀)) / (X - x₀)
      // W₁(X) = H₁(X) / (X - x₁) = (P₀(X) - P₀(x₁)) + v(P₁(X) - P₁(x₁)) + v²(P₂(X) - P₂(x₁)) / (X - x₁)
      // W₂(X) = H₂(X) / (X - x₂) = (P₀(X) - P₀(x₂)) + v(P₁(X) - P₁(x₂)) + v²(P₂(X) - P₂(x₂)) + v³(P₃(X) - P₃(x₂)) / (X - x₂)
      // W₃(X) = H₃(X) / (X - x₃) = (P₃(X) - P₃(x₃)) / (X - x₃)
      // W₄(X) = H₄(X) / (X - x₄) = (P₄(X) - P₄(x₄)) / (X - x₄)
      // clang-format on
      std::vector<Poly> low_degree_extensions;
      Poly w = grouped_poly_openings_vec[i].CreateCombinedLowDegreeExtensions(
          v, low_degree_extensions);
      if (!this->Commit(w, i)) return false;
    }

    // Commit all the Wᵢ(X).
    std::vector<Commitment> commitments = this->GetBatchCommitments();
    for (const Commitment& commitment : commitments) {
      if (!writer->WriteToProof(commitment)) return false;
    }
    return true;
  }

  template <typename Container>
  [[nodiscard]] bool DoVerifyOpeningProof(
      const Container& poly_openings,
      TranscriptReader<Commitment>* reader) const {
    using G1JacobianPoint = math::JacobianPoint<typename G1Point::Curve>;

    Field v = reader->SqueezeChallenge();
    VLOG(2) << "GWC(v): " << v.ToHexString(true);

    PolynomialOpeningGrouper<Poly, Commitment> grouper;
    grouper.GroupBySinglePoint(poly_openings);

    // Group |poly_openings| to |grouped_poly_openings_vec|.
    // {x₀, [C₀, C₁, C₂]}
    // {x₁, [C₀, C₁, C₂]}
    // {x₂, [C₀, C₁, C₂, C₃]}
    // {x₃, [C₃]}
    // {x₄, [C₄]}
    const std::vector<GroupedPolynomialOpenings<Poly, Commitment>>&
        grouped_poly_openings_vec = grouper.grouped_poly_openings_vec();

    // |commitments| = [[W₀(𝜏)]₁, [W₁(𝜏)]₁, [W₂(𝜏)]₁, [W₃(𝜏)]₁, [W₄(𝜏)]₁]
    std::vector<Commitment> commitments;
    commitments.reserve(grouped_poly_openings_vec.size());
    for (size_t i = 0; i < grouped_poly_openings_vec.size(); ++i) {
      Commitment commitment;
      if (!reader->ReadFromProof(&commitment)) return false;
      commitments.push_back(std::move(commitment));
    }

    Field u = reader->SqueezeChallenge();
    VLOG(2) << "GWC(u): " << u.ToHexString(true);

    Field opening_multi = Field::Zero();
    G1JacobianPoint commitment_multi = G1JacobianPoint::Zero();
    G1JacobianPoint witness_with_aux = G1JacobianPoint::Zero();
    G1JacobianPoint witness = G1JacobianPoint::Zero();
    Field power_of_u = Field::One();
    for (size_t i = 0; i < grouped_poly_openings_vec.size(); ++i) {
      Field opening_batch = Field::Zero();
      G1JacobianPoint commitment_batch = G1JacobianPoint::Zero();

      const std::vector<PolynomialOpenings<Poly, Commitment>>&
          poly_openings_vec = grouped_poly_openings_vec[i].poly_openings_vec;
      for (const PolynomialOpenings<Poly, Commitment>& poly_openings :
           base::Reversed(poly_openings_vec)) {
        // |opening_batch₀| = P₀(x₀) + vP₁(x₀) + v²P₂(x₀)
        // |opening_batch₁| = P₀(x₁) + vP₁(x₁) + v²P₂(x₁)
        // |opening_batch₂| = P₀(x₂) + vP₁(x₂) + v²P₂(x₂) + v³P₃(x₂)
        // |opening_batch₃| = P₃(x₃)
        // |opening_batch₄| = P₄(x₄)
        opening_batch *= v;
        opening_batch += poly_openings.openings[0];
        // |commitment_batch₀| = C₀ + vC₁ + v²C₂
        // |commitment_batch₁| = C₀ + vC₁ + v²C₂
        // |commitment_batch₂| = C₀ + vC₁ + v²C₂ + v³C₃
        // |commitment_batch₃| = C₃
        // |commitment_batch₄| = C₄
        commitment_batch *= v;
        commitment_batch += *poly_openings.poly_oracle;
      }

      // |commitment_multi| = Cₘᵤₗₜ = C₀ + vC₁ + v²C₂ +
      //                              u(C₀ + vC₁ + v²C₂) +
      //                              u²(C₀ + vC₁ + v²C₂ + v³C₃) +
      //                              u³C₃ +
      //                              u⁴C₄
      commitment_batch *= power_of_u;
      commitment_multi += commitment_batch;
      // |opening_multi| = Oₘᵤₗₜ = P₀(x₀) + vP₁(x₀) + v²P₂(x₀) +
      //                           u(P₀(x₁) + vP₁(x₁) + v²P₂(x₁)) +
      //                           u²(P₀(x₂) + vP₁(x₂) + v²P₂(x₂) + v³P₃(x₂)) +
      //                           u³P₃(x₃) +
      //                           u⁴P₄(x₄)
      opening_batch *= power_of_u;
      opening_multi += opening_batch;

      // clang-format off
      // |witness_with_aux| = Wₐᵤₓ = x₀[W₀(𝜏)]₁ + ux₁[W₁(𝜏)]₁ + u²x₂[W₂(𝜏)]₁ + u³x₃[W₃(𝜏)]₁ + u⁴x₄[W₄(𝜏)]₁
      // clang-format on
      witness_with_aux += power_of_u *
                          *grouped_poly_openings_vec[i].point_refs[0] *
                          commitments[i];
      // clang-format off
      // |witness| = W = [W₀(𝜏)]₁ + u[W₁(𝜏)]₁ + u²[W₂(𝜏)]₁ + u³[W₃(𝜏)]₁ + u⁴[W₄(𝜏)]₁
      // clang-format on
      witness += power_of_u * commitments[i];

      power_of_u *= u;
    }
    // clang-format off
    // e(W, [𝜏]₂) * e(Wₐᵤₓ + Cₘᵤₗₜ - [Oₘᵤₗₜ]₁, [-1]₂) ≟ gᴛ⁰
    // 𝜏(W₀(𝜏) + uW₁(𝜏) + u²W₂(𝜏) + u³W₃(𝜏) + u⁴W₄(𝜏)) - x₀W₀(𝜏) - ux₁W₁(𝜏) - u²x₂W₂(𝜏) - u³x₃W₃(𝜏) - u⁴x₄W₄(𝜏) -
    // P₀(𝜏) - vP₁(𝜏) - v²P₂(𝜏) - u(P₀(𝜏) + vP₁(𝜏) + v²P₂(𝜏)) - u²(P₀(𝜏) + vP₁(𝜏) + v²P₂(𝜏) + v³P₃(𝜏)) -
    // u³P₃(𝜏) - u⁴P₄(𝜏) + P₀(x₀) + vP₁(x₀) + v²P₂(x₀) + u(P₀(x₁) + vP₁(x₁) + v²P₂(x₁)) +
    // u²(P₀(x₂) + vP₁(x₂) + v²P₂(x₂) + v³P₃(x₂)) + u³P₃(x₃) + u⁴P₄(x₄) ≟ 0
    // (𝜏 - x₀)W₀(𝜏) + u(𝜏 - x₁)W₁(𝜏) + u²(𝜏 - x₂)W₂(𝜏) + u³(𝜏 - x₃)W₃(𝜏) + u⁴(𝜏 - x₄)W₄(𝜏) -
    // H₀(𝜏) - uH₁(𝜏) - u²H₂(𝜏) - u³H₃(𝜏) - u⁴H₄(𝜏) ≟ 0
    // clang-format on
    G1JacobianPoint g1_jacobian_arr[] = {
        witness, (witness_with_aux + commitment_multi -
                  opening_multi * G1JacobianPoint::Generator())};
    G1Point g1_arr[2];
    if (!G1JacobianPoint::BatchNormalize(g1_jacobian_arr, &g1_arr))
      return false;
    return math::Pairing<Curve>(g1_arr, g2_arr_).IsOne();
  }

  // KZGFamily methods
  [[nodiscard]] bool DoUnsafeSetupWithTau(size_t size,
                                          const Field& tau) override {
    s_g2_ = (G2Point::Generator() * tau).ToAffine();
    g2_arr_ = {
        G2Prepared::From(s_g2_),
        G2Prepared::From(-G2Point::Generator()),
    };
    return true;
  }

  G2Point s_g2_;
  std::array<G2Prepared, 2> g2_arr_;
};

template <typename Curve, size_t MaxDegree, typename _Commitment>
struct VectorCommitmentSchemeTraits<GWC<Curve, MaxDegree, _Commitment>> {
 public:
  constexpr static size_t kMaxSize = MaxDegree + 1;
  constexpr static bool kIsTransparent = false;
  constexpr static bool kSupportsBatchMode = true;

  using G1Point = typename Curve::G1Curve::AffinePoint;
  using Field = typename G1Point::ScalarField;
  using Commitment = _Commitment;
};

}  // namespace crypto

namespace base {

template <typename Curve, size_t MaxDegree, typename Commitment>
class Copyable<crypto::GWC<Curve, MaxDegree, Commitment>> {
 public:
  using G1Point = typename Curve::G1Curve::AffinePoint;
  using G2Point = typename Curve::G2Curve::AffinePoint;
  using KZG = crypto::KZG<G1Point, MaxDegree, Commitment>;
  using PCS = crypto::GWC<Curve, MaxDegree, Commitment>;

  static bool WriteTo(const PCS& pcs, Buffer* buffer) {
    return buffer->WriteMany(pcs.kzg(), pcs.s_g2());
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer, PCS* pcs) {
    KZG kzg;
    G2Point s_g2;
    if (!buffer.ReadMany(&kzg, &s_g2)) {
      return false;
    }

    *pcs = PCS(std::move(kzg), std::move(s_g2));
    return true;
  }

  static size_t EstimateSize(const PCS& pcs) {
    return base::EstimateSize(pcs.kzg()) + base::EstimateSize(pcs.s_g2());
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_COMMITMENTS_KZG_GWC_H_
