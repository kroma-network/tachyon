#ifndef VENDORS_CIRCOM_CIRCOMLIB_ZKEY_ZKEY_H_
#define VENDORS_CIRCOM_CIRCOMLIB_ZKEY_ZKEY_H_

#include <string.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "circomlib/base/modulus.h"
#include "circomlib/base/sections.h"
#include "circomlib/zkey/constraint_matrices.h"
#include "circomlib/zkey/proving_key.h"
#include "tachyon/base/auto_reset.h"
#include "tachyon/base/buffer/endian_auto_reset.h"
#include "tachyon/base/buffer/read_only_buffer.h"
#include "tachyon/base/files/file_util.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"

namespace tachyon::circom {
namespace v1 {

template <typename Curve>
struct ZKey;

}  // namespace v1

template <typename Curve>
struct ZKey {
  using F = typename Curve::G1Curve::ScalarField;

  explicit ZKey(std::vector<uint8_t>&& data) : data(std::move(data)) {}
  virtual ~ZKey() = default;

  virtual uint32_t GetVersion() const = 0;

  virtual v1::ZKey<Curve>* ToV1() { return nullptr; }

  virtual bool Read(const base::ReadOnlyBuffer& buffer) = 0;

  virtual ProvingKey<Curve> GetProvingKey() const = 0;
  virtual ConstraintMatrices<F> TakeConstraintMatrices() && = 0;

  std::vector<uint8_t> data;
};

constexpr char kZkeyMagic[4] = {'z', 'k', 'e', 'y'};

// Return nullptr if the parser failed to parse.
template <typename Curve>
std::unique_ptr<ZKey<Curve>> ParseZKey(const base::FilePath& path) {
  std::optional<std::vector<uint8_t>> zkey_data = base::ReadFileToBytes(path);
  if (!zkey_data.has_value()) {
    LOG(ERROR) << "Failed to read file: " << path.value();
    return nullptr;
  }

  base::ReadOnlyBuffer buffer(zkey_data->data(), zkey_data->size());
  buffer.set_endian(base::Endian::kLittle);
  char magic[4];
  uint32_t version;
  if (!buffer.ReadMany(magic, &version)) return nullptr;
  if (memcmp(magic, kZkeyMagic, 4) != 0) {
    LOG(ERROR) << "Invalid magic: " << magic;
    return nullptr;
  }
  std::unique_ptr<ZKey<Curve>> zkey;
  if (version == 1) {
    zkey.reset(new v1::ZKey<Curve>(std::move(zkey_data).value()));
    CHECK(zkey->ToV1()->Read(buffer));
  } else {
    LOG(ERROR) << "Invalid version: " << version;
    return nullptr;
  }
  return zkey;
}

namespace v1 {

enum class ZKeySectionType : uint32_t {
  kHeader = 0x1,
  kHeaderGroth = 0x2,
  kIC = 0x3,
  kCoefficients = 0x4,
  kPointsA1 = 0x5,
  kPointsB1 = 0x6,
  kPointsB2 = 0x7,
  kPointsC1 = 0x8,
  kPointsH1 = 0x9,
  kContribution = 0xa,
};

std::string_view ZKeySectionTypeToString(ZKeySectionType type);

struct ZKeyHeaderSection {
  uint32_t prover_type;

  bool operator==(const ZKeyHeaderSection& other) const {
    return prover_type == other.prover_type;
  }
  bool operator!=(const ZKeyHeaderSection& other) const {
    return prover_type != other.prover_type;
  }

  bool Read(const base::ReadOnlyBuffer& buffer) {
    base::EndianAutoReset reset(buffer, base::Endian::kLittle);
    if (!buffer.Read(&prover_type)) return false;
    if (prover_type != 1) {
      LOG(ERROR) << "Unknown prover_type: " << prover_type;
      return false;
    }
    return true;
  }

  std::string ToString() const {
    return absl::Substitute("{prover_type: $0}", prover_type);
  }
};

template <typename Curve>
struct ZKeyHeaderGrothSection {
  Modulus q;
  Modulus r;
  uint32_t num_vars;
  uint32_t num_public_inputs;
  uint32_t domain_size;
  VerifyingKey<Curve> vkey;

  bool operator==(const ZKeyHeaderGrothSection& other) const {
    return q == other.q && r == other.r && num_vars == other.num_vars &&
           num_public_inputs == other.num_public_inputs &&
           domain_size == other.domain_size && vkey == other.vkey;
  }
  bool operator!=(const ZKeyHeaderGrothSection& other) const {
    return !operator==(other);
  }

  bool Read(const base::ReadOnlyBuffer& buffer) {
    base::EndianAutoReset reset(buffer, base::Endian::kLittle);
    if (!q.Read(buffer)) return false;
    if (!r.Read(buffer)) return false;
    if (!buffer.ReadMany(&num_vars, &num_public_inputs, &domain_size))
      return false;
    return vkey.Read(buffer);
  }

  std::string ToString() const {
    return absl::Substitute(
        "{q: $0, r: $1, num_vars: $2, num_public_inputs: $3, domain_size: $4, "
        "vkey: $5}",
        q.ToString(), r.ToString(), num_vars, num_public_inputs, domain_size,
        vkey.ToString());
  }
};

template <typename T>
struct CommitmentsSection {
  absl::Span<T> commitments;

  bool operator==(const CommitmentsSection& other) const {
    return commitments == other.commitments;
  }
  bool operator!=(const CommitmentsSection& other) const {
    return commitments != other.commitments;
  }

  bool Read(const base::ReadOnlyBuffer& buffer, uint32_t num_commitments) {
    T* ptr;
    if (!buffer.ReadPtr(&ptr, num_commitments)) return false;
    commitments = {ptr, num_commitments};
    return true;
  }

  // NOTE(chokobole): the fields are represented in montgomery form.
  std::string ToString() const { return base::ContainerToString(commitments); }
};

template <typename C>
using ICSection = CommitmentsSection<C>;
template <typename C>
using PointsA1Section = CommitmentsSection<C>;
template <typename C>
using PointsB1Section = CommitmentsSection<C>;
template <typename C>
using PointsB2Section = CommitmentsSection<C>;
template <typename C>
using PointsC1Section = CommitmentsSection<C>;
template <typename C>
using PointsH1Section = CommitmentsSection<C>;

template <typename F>
struct CoefficientsSection {
  std::vector<std::vector<Cell<F>>> a;
  std::vector<std::vector<Cell<F>>> b;
  uint32_t max_constraint;

  bool operator==(const CoefficientsSection& other) const {
    return a == other.a && b == other.b;
  }
  bool operator!=(const CoefficientsSection& other) const {
    return !operator==(other);
  }

  bool Read(const base::ReadOnlyBuffer& buffer, uint32_t domain_size) {
    base::EndianAutoReset reset(buffer, base::Endian::kLittle);
    uint32_t num_coefficients;
    if (!buffer.Read(&num_coefficients)) return false;
    a.resize(domain_size);
    b.resize(domain_size);
    max_constraint = 0;
    for (uint32_t i = 0; i < num_coefficients; ++i) {
      Cell<F> cell;
      uint32_t matrix, constraint;
      if (!buffer.ReadMany(&matrix, &constraint, &cell.signal,
                           &cell.coefficient))
        return false;
      max_constraint = std::max(constraint, max_constraint);
      if (matrix == 0) {
        a[constraint].push_back(std::move(cell));
      } else {
        b[constraint].push_back(std::move(cell));
      }
    }
    return true;
  }

  std::string ToString() const {
    return absl::Substitute("{a: $0, b: $1}", base::Container2DToString(a),
                            base::Container2DToString(b));
  }
};

template <typename Curve>
struct ZKey : public circom::ZKey<Curve> {
  using G1AffinePoint = typename Curve::G1Curve::AffinePoint;
  using G2AffinePoint = typename Curve::G2Curve::AffinePoint;
  using F = typename G1AffinePoint::ScalarField;

  ZKeyHeaderSection header;
  ZKeyHeaderGrothSection<Curve> header_groth;
  ICSection<G1AffinePoint> ic;
  CoefficientsSection<F> coefficients;
  PointsA1Section<G1AffinePoint> points_a1;
  PointsB1Section<G1AffinePoint> points_b1;
  PointsB2Section<G2AffinePoint> points_b2;
  PointsC1Section<G1AffinePoint> points_c1;
  PointsH1Section<G1AffinePoint> points_h1;

  explicit ZKey(std::vector<uint8_t>&& data)
      : circom::ZKey<Curve>(std::move(data)) {}

  // circom::ZKey methods
  uint32_t GetVersion() const override { return 1; }
  ZKey<Curve>* ToV1() override { return this; }

  bool Read(const base::ReadOnlyBuffer& buffer) override {
    using BaseField = typename G1AffinePoint::BaseField;

    base::AutoReset<bool> auto_reset(&base::Copyable<F>::s_is_in_montgomery,
                                     true);
    base::AutoReset<bool> auto_reset2(
        &base::Copyable<BaseField>::s_is_in_montgomery, true);

    Sections<ZKeySectionType> sections(buffer, &ZKeySectionTypeToString);
    if (!sections.Read()) return false;

    if (!sections.MoveTo(ZKeySectionType::kHeader)) return false;
    if (!header.Read(buffer)) return false;

    if (!sections.MoveTo(ZKeySectionType::kHeaderGroth)) return false;
    if (!header_groth.Read(buffer)) return false;
    uint32_t num_vars = header_groth.num_vars;
    uint32_t num_public_inputs = header_groth.num_public_inputs;
    uint32_t domain_size = header_groth.domain_size;

    if (!sections.MoveTo(ZKeySectionType::kIC)) return false;
    if (!ic.Read(buffer, num_public_inputs + 1)) return false;

    if (!sections.MoveTo(ZKeySectionType::kCoefficients)) return false;
    if (!coefficients.Read(buffer, domain_size)) return false;

    if (!sections.MoveTo(ZKeySectionType::kPointsA1)) return false;
    if (!points_a1.Read(buffer, num_vars)) return false;

    if (!sections.MoveTo(ZKeySectionType::kPointsB1)) return false;
    if (!points_b1.Read(buffer, num_vars)) return false;

    if (!sections.MoveTo(ZKeySectionType::kPointsB2)) return false;
    if (!points_b2.Read(buffer, num_vars)) return false;

    if (!sections.MoveTo(ZKeySectionType::kPointsC1)) return false;
    if (!points_c1.Read(buffer, num_vars - num_public_inputs - 1)) return false;

    if (!sections.MoveTo(ZKeySectionType::kPointsH1)) return false;
    if (!points_h1.Read(buffer, domain_size)) return false;
    return true;
  }

  ProvingKey<Curve> GetProvingKey() const override {
    return {
        header_groth.vkey,     ic.commitments,        points_a1.commitments,
        points_b1.commitments, points_b2.commitments, points_c1.commitments,
        points_h1.commitments,
    };
  }

  ConstraintMatrices<F> TakeConstraintMatrices() && override {
    size_t num_constraints =
        coefficients.max_constraint - header_groth.num_public_inputs;
    if (coefficients.a.size() > num_constraints) {
      coefficients.a.resize(num_constraints);
    }
    if (coefficients.b.size() > num_constraints) {
      coefficients.b.resize(num_constraints);
    }

    size_t a_num_non_zero =
        std::accumulate(coefficients.a.begin(), coefficients.a.end(), 0,
                        [](size_t acc, const std::vector<Cell<F>>& cells) {
                          return acc + cells.size();
                        });
    size_t b_num_non_zero =
        std::accumulate(coefficients.b.begin(), coefficients.b.end(), 0,
                        [](size_t acc, const std::vector<Cell<F>>& cells) {
                          return acc + cells.size();
                        });

    return {
        header_groth.num_public_inputs + 1,
        header_groth.num_vars - header_groth.num_public_inputs - 1,
        num_constraints,

        a_num_non_zero,
        b_num_non_zero,

        std::move(coefficients.a),
        std::move(coefficients.b),
    };
  }

  std::string ToString() const {
    return absl::Substitute(
        "{header: $0, header_groth: $1, ic: $2, coefficients: $3, points_a1: "
        "$4, points_b1: $5, points_b2: $6, points_c1: $7, points_h1: $8}",
        header.ToString(), header_groth.ToString(), ic.ToString(),
        coefficients.ToString(), points_a1.ToString(), points_b1.ToString(),
        points_b2.ToString(), points_c1.ToString(), points_h1.ToString());
  }
};

}  // namespace v1
}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_ZKEY_ZKEY_H_
