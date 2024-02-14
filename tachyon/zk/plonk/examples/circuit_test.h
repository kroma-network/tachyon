#ifndef TACHYON_ZK_PLONK_EXAMPLES_CIRCUIT_TEST_H_
#define TACHYON_ZK_PLONK_EXAMPLES_CIRCUIT_TEST_H_

#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"

#include "tachyon/zk/lookup/lookup_pair.h"
#include "tachyon/zk/plonk/halo2/bn254_shplonk_prover_test.h"

namespace tachyon::zk::plonk::halo2 {

class CircuitTest : public BN254SHPlonkProverTest {
 protected:
  struct Point {
    std::string_view x;
    std::string_view y;
  };

  static Commitment CreateCommitment(const Point& point) {
    return Commitment(math::bn254::Fq::FromHexString(point.x),
                      math::bn254::Fq::FromHexString(point.y));
  }

  static std::vector<Commitment> CreateCommitments(
      const std::vector<Point>& points) {
    return base::Map(points, &CreateCommitment);
  }

  static std::vector<LookupPair<Commitment>> CreateLookupPermutedCommitments(
      const std::vector<Point>& input_points,
      const std::vector<Point>& table_points) {
    std::vector<LookupPair<Commitment>> lookup_pairs;
    return base::Map(
        input_points, [&table_points](size_t i, const Point& input_point) {
          return LookupPair<Commitment>(CreateCommitment(input_point),
                                        CreateCommitment(table_points[i]));
        });
  }

  static Evals CreateColumn(const std::vector<std::string_view>& column) {
    std::vector<F> evaluations = base::Map(
        column, [](std::string_view coeff) { return F::FromHexString(coeff); });
    return Evals(std::move(evaluations));
  }

  static std::vector<Evals> CreateColumns(
      const std::vector<std::vector<std::string_view>>& columns) {
    return base::Map(columns, &CreateColumn);
  }

  static RationalEvals CreateRationalColumn(
      const std::vector<std::string_view>& column) {
    std::vector<math::RationalField<F>> evaluations =
        base::Map(column, [](std::string_view coeff) {
          return math::RationalField<F>(F::FromHexString(coeff));
        });
    return RationalEvals(std::move(evaluations));
  }

  static std::vector<RationalEvals> CreateRationalColumns(
      const std::vector<std::vector<std::string_view>>& columns) {
    return base::Map(columns, &CreateRationalColumn);
  }

  static Poly CreatePoly(const std::vector<std::string_view>& poly) {
    std::vector<F> coefficients = base::Map(
        poly, [](std::string_view coeff) { return F::FromHexString(coeff); });
    return Poly(math::UnivariateDenseCoefficients<F, kMaxDegree>(
        std::move(coefficients)));
  }

  static std::vector<Poly> CreatePolys(
      const std::vector<std::vector<std::string_view>>& polys) {
    return base::Map(polys, &CreatePoly);
  }

  static std::vector<F> CreateEvals(
      const std::vector<std::string_view>& evals) {
    return base::Map(
        evals, [](std::string_view eval) { return F::FromHexString(eval); });
  }

  static std::vector<std::optional<F>> CreateOptionalEvals(
      const std::vector<std::string_view>& evals) {
    return base::Map(evals, [](std::string_view eval) {
      if (eval.empty()) return std::optional<F>();
      return std::optional<F>(F::FromHexString(eval));
    });
  }

  static base::Buffer CreateBufferWithProof(absl::Span<uint8_t> proof) {
    return {proof.data(), proof.size()};
  }
};

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_EXAMPLES_CIRCUIT_TEST_H_
