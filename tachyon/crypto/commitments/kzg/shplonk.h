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
#include "tachyon/math/elliptic_curves/pairing/pairing.h"

namespace tachyon {
namespace zk {

template <typename CurveTy, size_t MaxDegree, size_t MaxExtensionDegree,
          typename _Commitment = typename math::Pippenger<
              typename CurveTy::G1Curve::AffinePointTy>::Bucket>
class SHPlonkExtension;

}  // namespace zk

namespace crypto {

template <typename CurveTy, size_t MaxDegree,
          typename Commitment = typename math::Pippenger<
              typename CurveTy::G1Curve::AffinePointTy>::Bucket>
class SHPlonk : public UnivariatePolynomialCommitmentScheme<
                    SHPlonk<CurveTy, MaxDegree, Commitment>>,
                public KZGFamily<typename CurveTy::G1Curve::AffinePointTy,
                                 MaxDegree, Commitment> {
 public:
  using Base = UnivariatePolynomialCommitmentScheme<
      SHPlonk<CurveTy, MaxDegree, Commitment>>;
  using G1PointTy = typename CurveTy::G1Curve::AffinePointTy;
  using G2PointTy = typename CurveTy::G2Curve::AffinePointTy;
  using G2Prepared = typename CurveTy::G2Prepared;
  using Fp12Ty = typename CurveTy::Fp12Ty;
  using Field = typename Base::Field;
  using Poly = typename Base::Poly;
  using Point = typename Poly::Point;
  using PointDeepRef = base::DeepRef<const Point>;

  SHPlonk() = default;
  explicit SHPlonk(KZG<G1PointTy, MaxDegree, Commitment>&& kzg)
      : KZGFamily<G1PointTy, MaxDegree, Commitment>(std::move(kzg)) {}

 private:
  friend class VectorCommitmentScheme<SHPlonk<CurveTy, MaxDegree, Commitment>>;
  friend class UnivariatePolynomialCommitmentScheme<
      SHPlonk<CurveTy, MaxDegree, Commitment>>;
  template <typename, size_t, size_t, typename>
  friend class zk::SHPlonkExtension;

  // Set 𝜏G₂
  void SetTauG2(const G2PointTy& tau_g2) { tau_g2_ = tau_g2; }

  // UnivariatePolynomialCommitmentScheme methods
  template <typename ContainerTy>
  [[nodiscard]] bool DoCreateOpeningProof(
      const ContainerTy& poly_openings,
      TranscriptWriter<Commitment>* writer) const {
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

    Field y = writer->SqueezeChallenge();

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

    Field v = writer->SqueezeChallenge();

    // Create a linear combination of polynomials [H₀(X), H₁(X), H₂(X)] with
    // with |v|.
    // H(X) = H₀(X) + vH₁(X) + v²H₂(X)
    Poly& h_poly = Poly::LinearizeInPlace(h_polys, v);

    // Commit H(X)
    Commitment h;
    if (!this->Commit(h_poly, &h)) return false;

    if (!writer->WriteToProof(h)) return false;
    Field u = writer->SqueezeChallenge();

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
          for (PointDeepRef point_ref : grouped_poly_openings.point_refs) {
            diffs.erase(point_ref);
          }

          std::vector<Point> diffs_vec = base::Map(
              diffs, [](PointDeepRef point_ref) { return *point_ref; });
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
                Poly poly = *poly_openings.poly_oracle;
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
    return writer->WriteToProof(q);
  }

  template <typename ContainerTy>
  [[nodiscard]] bool DoVerifyOpeningProof(
      const ContainerTy& poly_openings,
      TranscriptReader<Commitment>* reader) const {
    using G1JacobianPointTy = typename G1PointTy::JacobianPointTy;

    Field y = reader->SqueezeChallenge();
    Field v = reader->SqueezeChallenge();

    Commitment h;
    if (!reader->ReadFromProof(&h)) return false;

    Field u = reader->SqueezeChallenge();

    Commitment q;
    if (!reader->ReadFromProof(&q)) return false;

    PolynomialOpeningGrouper<Poly, Commitment> grouper;
    grouper.GroupByPolyAndPoints(poly_openings);

    // Group |poly_openings| to |grouped_poly_openings_vec|.
    // {[C₀, C₁, C₂], [x₀, x₁, x₂]}
    // {[C₃], [x₂, x₃]}
    // {[C₄], [x₄]}
    const std::vector<GroupedPolynomialOpenings<Poly, Commitment>>&
        grouped_poly_openings_vec = grouper.grouped_poly_openings_vec();
    const absl::btree_set<PointDeepRef>& super_point_set =
        grouper.super_point_set();

    Field first_z_diff_inverse = Field::Zero();
    Field first_z = Field::Zero();

    std::vector<G1JacobianPointTy> l_commitments;
    l_commitments.reserve(grouped_poly_openings_vec.size());
    size_t i = 0;
    for (const auto& [poly_openings_vec, point_refs] :
         grouped_poly_openings_vec) {
      // |commitments[0]| = [C₀, C₁, C₂]
      // |commitments[1]| = [C₃]
      // |commitments[2]| = [C₄]
      std::vector<Commitment> commitments = base::Map(
          poly_openings_vec,
          [](const PolynomialOpenings<Poly, Commitment>& poly_openings) {
            return *poly_openings.poly_oracle;
          });
      // |points[0]| = [x₀, x₁, x₂]
      // |points[1]| = [x₂, x₃]
      // |points[2]| = [x₄]
      std::vector<Point> points = base::Map(
          point_refs, [](const PointDeepRef& point_ref) { return *point_ref; });
      // |diffs[0]| = [x₃, x₄]
      // |diffs[1]| = [x₀, x₁, x₄]
      // |diffs[2]| = [x₀, x₁, x₂, x₃]
      std::vector<Point> diffs;
      diffs.reserve(point_refs.size());
      for (const PointDeepRef& point_ref : super_point_set) {
        if (std::find(point_refs.begin(), point_refs.end(), point_ref) ==
            point_refs.end()) {
          diffs.push_back(*point_ref);
        }
      }

      // z_diff_0 = (u - x₃)(u - x₄)
      // z_diff_1 = (u - x₀)(u - x₁)(u - x₄)
      // z_diff_2 = (u - x₀)(u - x₁)(u - x₂)(u - x₃)
      Point z_diff_i = Poly::EvaluateVanishingPolyByRoots(std::move(diffs), u);
      if (i == 0) {
        // z₀ = (u - x₀)(u - x₁)(u - x₂)
        first_z = Poly::EvaluateVanishingPolyByRoots(std::move(points), u);
        // (u - x₃)(u - x₄)⁻¹
        first_z_diff_inverse = z_diff_i.InverseInPlace();
        z_diff_i = Field::One();
      } else {
        // z_diff_1 = (u - x₀)(u - x₁)(u - x₄)/(u - x₃)(u - x₄)
        // z_diff_2 = (u - x₀)(u - x₁)(u - x₂)(u - x₃)/(u - x₃)(u - x₄)
        z_diff_i *= first_z_diff_inverse;
      }

      // For example in i = 0:
      // [R₀(u)]₁ = R₀(u) * G₁
      // [R₁(u)]₁ = R₁(u) * G₁
      // [R₂(u)]₁ = R₂(u) * G₁
      // r_commitments = [R₀(u) * G₁, R₁(u) * G₁, R₂(u) * G₁]
      std::vector<G1JacobianPointTy> r_commitments = base::Map(
          poly_openings_vec,
          [&points,
           &u](const PolynomialOpenings<Poly, Commitment>& poly_openings) {
            Poly r_i;
            CHECK(math::LagrangeInterpolate(points, poly_openings.openings,
                                            &r_i));
            Field r = r_i.Evaluate(u);
            return G1PointTy::Generator().ScalarMul(r);
          });

      // [L₀]₁ = (C₀ - [R₀(u)]₁) + y(C₁ - [R₁(u)]₁) + y²(C₂ - [R₂(u)]₁)
      G1JacobianPointTy l_i = commitments.back() - r_commitments.back();
      if (commitments.size() > 1) {
        for (size_t j = commitments.size() - 2; j != SIZE_MAX; --j) {
          l_i *= y;
          l_i += (commitments[j] - r_commitments[j]);
        }
      }

      // [L₀]₁ *= 1
      // [L₁]₁ *= (u - x₀)(u - x₁)(u - x₄)/(u - x₃)(u - x₄)
      // [L₂]₁ *= (u - x₀)(u - x₁)(u - x₂)(u - x₃)/(u - x₃)(u - x₄)
      l_i *= z_diff_i;
      l_commitments.push_back(l_i);
      ++i;
    }

    // [L₀]₁ + v[L₁]₁ + v²[L₂]₁
    G1JacobianPointTy linear_combination =
        l_commitments[l_commitments.size() - 1];
    if (l_commitments.size() > 1) {
      for (size_t j = l_commitments.size() - 2; j != SIZE_MAX; --j) {
        linear_combination *= v;
        linear_combination += l_commitments[j];
      }
    }

    // lhs_g1 = [L₀]₁ + v[L₁]₁ + v²[L₂]₁ - z₀[H]₁ + u[Q]₁
    // lhs_g2 = G₂
    G1JacobianPointTy lhs = linear_combination;

    lhs -= (h * first_z);
    lhs += (q * u);

    std::vector<G1PointTy> lhs_g1 = {lhs.ToAffine()};
    std::vector<G2Prepared> lhs_g2 = {
        CurveTy::G2Prepared::From(G2PointTy::Generator())};
    Fp12Ty lhs_pairing =
        math::Pairing<CurveTy>(std::move(lhs_g1), std::move(lhs_g2));

    // rhs_g1 = [Q]₁
    // rhs_g2 = 𝜏G₂
    std::vector<G1PointTy> rhs_g1 = {q};
    std::vector<G2Prepared> rhs_g2 = {CurveTy::G2Prepared::From(tau_g2_)};
    Fp12Ty rhs_pairing =
        math::Pairing<CurveTy>(std::move(rhs_g1), std::move(rhs_g2));

    // e(lhs_g1, rhs_g2) == e(rhs_g1, lhs_g2)
    // lhs: e(G₁, G₂)^([L₀]₁ + v[L₁]₁ + v²[L₂]₁ - z₀[H]₁ + u[Q]₁)
    // rhs: e(G₁, G₂)^(𝜏[Q]₁)
    // [L₀]₁ + v[L₁]₁ + v²[L₂]₁ - z₀[H]₁ + u[Q]₁ ?= 𝜏[Q]₁
    // [L₀]₁ + v[L₁]₁ + v²[L₂]₁ - z₀[H]₁ ?= (𝜏 - u)[Q]₁
    return lhs_pairing == rhs_pairing;
  }

  // KZGFamily methods
  [[nodiscard]] bool DoUnsafeSetupWithTau(size_t size,
                                          const Field& tau) override {
    tau_g2_ = G2PointTy::Generator().ScalarMul(tau).ToAffine();
    return true;
  }

  G2PointTy tau_g2_;
};

template <typename CurveTy, size_t MaxDegree, typename _Commitment>
struct VectorCommitmentSchemeTraits<SHPlonk<CurveTy, MaxDegree, _Commitment>> {
 public:
  constexpr static size_t kMaxSize = MaxDegree + 1;
  constexpr static bool kIsTransparent = false;

  using G1PointTy = typename CurveTy::G1Curve::AffinePointTy;
  using Field = typename G1PointTy::ScalarField;
  using Commitment = _Commitment;
};

}  // namespace crypto
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_COMMITMENTS_KZG_SHPLONK_H_
