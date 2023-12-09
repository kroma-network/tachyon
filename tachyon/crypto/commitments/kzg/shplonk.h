// Copyright 2020-2022 The Electric Coin Company
// Copyright 2022 The Halo2 developers
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.halo2 and the LICENCE-APACHE.halo2
// file.

#ifndef TACHYON_CRYPTO_COMMITMENTS_KZG_SHPLONK_H_
#define TACHYON_CRYPTO_COMMITMENTS_KZG_SHPLONK_H_

#include <utility>
#include <vector>

#include "tachyon/crypto/commitments/kzg/kzg_family.h"
#include "tachyon/crypto/commitments/polynomial_openings.h"
#include "tachyon/crypto/commitments/univariate_polynomial_commitment_scheme.h"
#include "tachyon/crypto/transcripts/transcript.h"

namespace tachyon {
namespace zk {

template <typename G1PointTy, typename G2PointTy, size_t MaxDegree,
          size_t MaxExtensionDegree,
          typename _Commitment = typename math::Pippenger<G1PointTy>::Bucket>
class SHPlonkExtension;

}  // namespace zk

namespace crypto {

template <typename C>
class SHPlonkProof {
 public:
  SHPlonkProof() = default;
  SHPlonkProof(const C& h, const C& q) : h_(h), q_(q) {}
  SHPlonkProof(C&& h, C&& q) : h_(std::move(h)), q_(std::move(q)) {}

  const C& h() const { return h_; }
  const C& q() const { return q_; }

 private:
  C h_;
  C q_;
};

template <typename G1PointTy, typename G2PointTy, size_t MaxDegree,
          typename Commitment = typename math::Pippenger<G1PointTy>::Bucket>
class SHPlonk : public UnivariatePolynomialCommitmentScheme<
                    SHPlonk<G1PointTy, G2PointTy, MaxDegree, Commitment>>,
                public KZGFamily<G1PointTy, MaxDegree, Commitment> {
 public:
  using Base = UnivariatePolynomialCommitmentScheme<
      SHPlonk<G1PointTy, G2PointTy, MaxDegree, Commitment>>;
  using Field = typename Base::Field;
  using Poly = typename Base::Poly;

  SHPlonk() = default;
  SHPlonk(KZG<G1PointTy, MaxDegree, Commitment>&& kzg,
          Transcript<Commitment>* transcript)
      : KZGFamily<G1PointTy, MaxDegree, Commitment>(std::move(kzg)),
        transcript_(transcript) {}

 private:
  friend class VectorCommitmentScheme<
      SHPlonk<G1PointTy, G2PointTy, MaxDegree, Commitment>>;
  friend class UnivariatePolynomialCommitmentScheme<
      SHPlonk<G1PointTy, G2PointTy, MaxDegree, Commitment>>;
  template <typename, typename, size_t, size_t, typename>
  friend class zk::SHPlonkExtension;

  // UnivariatePolynomialCommitmentScheme methods
  template <typename ContainerTy>
  [[nodiscard]] bool DoCreateOpeningProof(
      const ContainerTy& poly_openings, SHPlonkProof<Commitment>* proof) const {
    using Point = typename Poly::Point;
    using PointDeepRef = base::DeepRef<const Point>;

    TranscriptWriter<Commitment>* writer = transcript_->ToWriter();

    PolynomialOpeningGrouper<Poly> grouper;
    grouper.GroupByPolyAndPoints(poly_openings);

    // Group |poly_openings| to |grouped_poly_openings_vec|.
    // {[P₀, P₁, P₂], [x₀, x₁, x₂]}
    // {[P₃], [x₂, x₃]}
    // {[P₄], [x₄]}
    const std::vector<GroupedPolynomialOpenings<Poly>>&
        grouped_poly_openings_vec = grouper.grouped_poly_openings_vec();
    const absl::btree_set<PointDeepRef>& super_point_set =
        grouper.super_point_set();

    Field y = writer->SqueezeChallengeAsScalar();

    // Create [H₀(X), H₁(X), H₂(X)].
    // clang-format off
    // H₀(X) = ((P₀(X) - R₀(X)) + y(P₁(X) - R₁(X)) + y²(P₂(X) - R₂(X))) / (X - x₀)(X - x₁)(X - x₂)
    // H₁(X) = ((P₃(X) - R₃(X)) / (X - x₂)(X - x₃)
    // H₂(X) = ((P₄(X) - R₄(X)) / (X - x₄)
    // clang-format on
    std::vector<std::vector<Poly>> low_degree_extensions_vec;
    low_degree_extensions_vec.resize(grouped_poly_openings_vec.size());
    std::vector<Poly> h_polys = base::Map(
        grouped_poly_openings_vec,
        [&y, &low_degree_extensions_vec](
            size_t i,
            const GroupedPolynomialOpenings<Poly>& grouped_poly_openings) {
          return grouped_poly_openings.CreateCombinedLowDegreeExtensions(
              y, low_degree_extensions_vec[i]);
        });

    Field v = writer->SqueezeChallengeAsScalar();

    // Create a linear combination of polynomials [H₀(X), H₁(X), H₂(X)] with
    // with |v|.
    // H(X) = H₀(X) + vH₁(X) + v²H₂(X)
    Poly& h_poly = Poly::LinearizeInPlace(h_polys, v);

    // Commit H(X)
    Commitment h;
    if (!this->Commit(h_poly, &h)) return false;

    CHECK(writer->WriteToProof(h));
    Field u = writer->SqueezeChallengeAsScalar();

    // Create [L₀(X), L₁(X), L₂(X)].
    // L₀(X) = z₀ * ((P₀(X) - R₀(u)) + y(P₁(X) - R₁(u)) + y²(P₂(X) - R₂(u)))
    // L₁(X) = z₁ * (P₃(X) - R₃(u))
    // L₂(X) = z₂ * (P₄(X) - R₄(u))
    Field first_z;
    std::vector<Poly> l_polys = base::Map(
        grouped_poly_openings_vec,
        [&y, &u, &first_z, &low_degree_extensions_vec, &super_point_set](
            size_t i,
            const GroupedPolynomialOpenings<Poly>& grouped_poly_openings) {
          absl::btree_set<PointDeepRef> diffs = super_point_set;
          for (PointDeepRef point : grouped_poly_openings.points) {
            diffs.erase(point);
          }

          std::vector<Point> diffs_vec =
              base::Map(diffs, [](PointDeepRef point) { return *point; });
          // calculate difference vanishing polynomial evaluation
          // z₀ = Z₀(u) = (u - x₃)(u - x₄)
          // z₁ = Z₁(u) = (u - x₀)(u - x₁)(u - x₄)
          // z₂ = Z₂(u) = (u - x₀)(u - x₁)(u - x₂)(u - x₃)
          Field z = Poly::EvaluateVanishingPolyByRoots(diffs_vec, u);
          if (i == 0) {
            first_z = z;
          }

          const std::vector<Poly>& low_degree_extensions =
              low_degree_extensions_vec[i];
          // L₀(X) = (P₀(X) - R₀(u)) + y(P₁(X) - R₁(u)) + y²(P₂(X) - R₂(u)))
          // L₁(X) = (P₃(X) - R₃(u))
          // L₂(X) = (P₄(X) - R₄(u))
          std::vector<Poly> polys = base::Map(
              grouped_poly_openings.poly_openings_vec,
              [&u, &low_degree_extensions](
                  size_t i, const PolynomialOpenings<Poly>& poly_openings) {
                Poly poly = *poly_openings.poly;
                *poly[0] -= low_degree_extensions[i].Evaluate(u);
                return poly;
              });

          Poly& l = Poly::LinearizeInPlace(polys, y);
          return l *= z;
        });

    // Create a linear combination of polynomials [L₀(X), L₁(X), L₂(X)] with
    // |v|.
    // L(X) = L₀(X) + vL₁(X) + v²L₂(X)
    Poly& l_poly = Poly::LinearizeInPlace(l_polys, v);

    // Zᴛ = [x₀, x₁, x₂, x₃, x₄]
    std::vector<Field> z_t =
        base::Map(super_point_set, [](const PointDeepRef& p) { return *p; });
    // Zᴛ(X) = (X - x₀)(X - x₁)(X - x₂)(X - x₃)(X - x₄)
    // Zᴛ(u) = (u - x₀)(u - x₁)(u - x₂)(u - x₃)(u - x₄)
    Field zt_eval = Poly::EvaluateVanishingPolyByRoots(z_t, u);

    // L(X) = L(X) - Zᴛ(u) * H(X)
    h_poly *= zt_eval;
    l_poly -= h_poly;

    // L(X) should be zero in X = |u|
    DCHECK(l_poly.Evaluate(u).IsZero());

    // Q(X) = L(X) / (X - u)
    Poly vanishing_poly = Poly::FromRoots(std::vector<Field>({u}));
    Poly& q_poly = l_poly /= vanishing_poly;

    // Normalize
    q_poly /= first_z;

    // Commit Q(X)
    Commitment q;
    if (!this->Commit(q_poly, &q)) return false;
    CHECK(writer->WriteToProof(q));

    *proof = {std::move(h), std::move(q)};
    return true;
  }

  // KZGFamily methods
  [[nodiscard]] bool DoUnsafeSetupWithTau(size_t size,
                                          const Field& tau) override {
    NOTIMPLEMENTED();
    return true;
  }

  // not owned
  Transcript<Commitment>* transcript_ = nullptr;
};

template <typename G1PointTy, typename G2PointTy, size_t MaxDegree,
          typename _Commitment>
struct VectorCommitmentSchemeTraits<
    SHPlonk<G1PointTy, G2PointTy, MaxDegree, _Commitment>> {
 public:
  constexpr static size_t kMaxSize = MaxDegree + 1;
  constexpr static bool kIsTransparent = false;

  using Field = typename G1PointTy::ScalarField;
  using Commitment = _Commitment;
};

}  // namespace crypto
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_COMMITMENTS_KZG_SHPLONK_H_
