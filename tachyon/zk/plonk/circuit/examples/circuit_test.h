#ifndef TACHYON_ZK_PLONK_CIRCUIT_EXAMPLES_CIRCUIT_TEST_H_
#define TACHYON_ZK_PLONK_CIRCUIT_EXAMPLES_CIRCUIT_TEST_H_

#include <utility>
#include <vector>

#include "tachyon/zk/plonk/halo2/prover_test.h"

namespace tachyon::zk {

class CircuitTest : public halo2::ProverTest {
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
};

}  // namespace tachyon::zk

#endif  // TACHYON_ZK_PLONK_CIRCUIT_EXAMPLES_CIRCUIT_TEST_H_
