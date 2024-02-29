#ifndef TACHYON_CRYPTO_COMMITMENTS_TEST_BN254_KZG_POLYNOMIAL_OPENINGS_H_
#define TACHYON_CRYPTO_COMMITMENTS_TEST_BN254_KZG_POLYNOMIAL_OPENINGS_H_

#include <string>
#include <utility>
#include <vector>

#include "tachyon/base/containers/container_util.h"
#include "tachyon/base/json/json.h"
#include "tachyon/base/ref.h"

namespace tachyon {
namespace crypto {

template <typename Poly, typename Commitment>
struct OwnedPolynomialOpening {
  using Point = typename Poly::Point;
  using Field = typename Poly::Field;

  Poly poly;
  Commitment commitment;
  Point point;
  Field opening;
};

template <typename Poly, typename Commitment>
struct OwnedPolynomialOpenings {
  using Point = typename Poly::Point;

  std::vector<OwnedPolynomialOpening<Poly, Commitment>> prover_openings;

  template <typename PCS>
  void Validate(const PCS& pcs) const {
    for (const OwnedPolynomialOpening<Poly, Commitment>& owned_opening :
         prover_openings) {
      CHECK_EQ(owned_opening.poly.Evaluate(owned_opening.point),
               owned_opening.opening);
      Commitment commitment;
      CHECK(pcs.Commit(owned_opening.poly, &commitment));
      CHECK_EQ(commitment, owned_opening.commitment);
    }
  }

  std::vector<PolynomialOpening<Poly>> CreateProverOpenings() const {
    return base::Map(
        prover_openings,
        [](const OwnedPolynomialOpening<Poly, Commitment>& owned_opening) {
          return PolynomialOpening<Poly>(
              base::Ref<const Poly>(&owned_opening.poly), owned_opening.point,
              owned_opening.opening);
        });
  }

  std::vector<PolynomialOpening<Poly, Commitment>> CreateVerifierOpenings()
      const {
    return base::Map(
        prover_openings,
        [](const OwnedPolynomialOpening<Poly, Commitment>& owned_opening) {
          return PolynomialOpening<Poly, Commitment>(
              base::Ref<const Commitment>(&owned_opening.commitment),
              owned_opening.point, owned_opening.opening);
        });
  }
};

}  // namespace crypto

namespace base {

template <typename Poly, typename Commitment>
class RapidJsonValueConverter<
    crypto::OwnedPolynomialOpening<Poly, Commitment>> {
 public:
  using Point = typename Poly::Point;
  using Field = typename Poly::Field;

  template <typename Allocator>
  static rapidjson::Value From(
      const crypto::OwnedPolynomialOpening<Poly, Commitment>& value,
      Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "poly", value.poly, allocator);
    AddJsonElement(object, "point", value.point, allocator);
    AddJsonElement(object, "opening", value.opening, allocator);
    AddJsonElement(object, "commitment", value.commitment, allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 crypto::OwnedPolynomialOpening<Poly, Commitment>* value,
                 std::string* error) {
    Poly poly;
    Commitment commitment;
    Point point;
    Field opening;
    if (!ParseJsonElement(json_value, "poly", &poly, error)) return false;
    if (!ParseJsonElement(json_value, "point", &point, error)) return false;
    if (!ParseJsonElement(json_value, "opening", &opening, error)) return false;
    if (!ParseJsonElement(json_value, "commitment", &commitment, error))
      return false;
    value->poly = std::move(poly);
    value->commitment = std::move(commitment);
    value->point = std::move(point);
    value->opening = std::move(opening);
    return true;
  }
};

template <typename Poly, typename Commitment>
class RapidJsonValueConverter<
    crypto::OwnedPolynomialOpenings<Poly, Commitment>> {
 public:
  template <typename Allocator>
  static rapidjson::Value From(
      const crypto::OwnedPolynomialOpenings<Poly, Commitment>& value,
      Allocator& allocator) {
    rapidjson::Value object(rapidjson::kObjectType);
    AddJsonElement(object, "prover_openings", value.prover_openings, allocator);
    return object;
  }

  static bool To(const rapidjson::Value& json_value, std::string_view key,
                 crypto::OwnedPolynomialOpenings<Poly, Commitment>* value,
                 std::string* error) {
    std::vector<crypto::OwnedPolynomialOpening<Poly, Commitment>>
        prover_openings;
    if (!ParseJsonElement(json_value, "prover_openings", &prover_openings,
                          error))
      return false;
    value->prover_openings = std::move(prover_openings);
    return true;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // TACHYON_CRYPTO_COMMITMENTS_TEST_BN254_KZG_POLYNOMIAL_OPENINGS_H_
