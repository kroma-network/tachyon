#ifndef VENDORS_CIRCOM_CIRCOMLIB_ZKEY_ZKEY_H_
#define VENDORS_CIRCOM_CIRCOMLIB_ZKEY_ZKEY_H_

#include <string>
#include <utility>
#include <vector>

#include "circomlib/base/sections.h"
#include "circomlib/zkey/cell.h"
#include "circomlib/zkey/verifying_key.h"
#include "tachyon/base/buffer/endian_auto_reset.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"

namespace tachyon::circom {
namespace v1 {

struct ZKey;

}  // namespace v1

struct ZKey {
  virtual ~ZKey() = default;

  virtual uint32_t GetVersion() const = 0;

  virtual v1::ZKey* ToV1() { return nullptr; }

  virtual bool Read(const base::ReadOnlyBuffer& buffer) = 0;
};

constexpr char kZkeyMagic[4] = {'z', 'k', 'e', 'y'};

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

struct ZKeyHeaderGrothSection {
  PrimeField q;
  PrimeField r;
  uint32_t num_vars;
  uint32_t num_public_inputs;
  uint32_t domain_size;
  VerifyingKey vkey;

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

    uint32_t field_size = q.bytes.size();
    return vkey.Read(buffer, field_size);
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
  std::vector<T> commitments;

  bool operator==(const CommitmentsSection& other) const {
    return commitments == other.commitments;
  }
  bool operator!=(const CommitmentsSection& other) const {
    return commitments != other.commitments;
  }

  bool Read(const base::ReadOnlyBuffer& buffer, uint32_t num_commitments,
            uint32_t field_size) {
    commitments.resize(num_commitments);
    for (uint32_t i = 0; i < num_commitments; ++i) {
      if (!commitments[i].Read(buffer, field_size)) return false;
    }
    return true;
  }

  template <typename F>
  void Normalize() {
    for (T& commitment : commitments) {
      commitment.template Normalize<F>();
    }
  }

  // NOTE(chokobole): the fields are represented in montgomery form.
  std::string ToString() const { return base::VectorToString(commitments); }
};

using ICSection = CommitmentsSection<G1AffinePoint>;
using PointsA1Section = CommitmentsSection<G1AffinePoint>;
using PointsB1Section = CommitmentsSection<G1AffinePoint>;
using PointsB2Section = CommitmentsSection<G2AffinePoint>;
using PointsC1Section = CommitmentsSection<G1AffinePoint>;
using PointsH1Section = CommitmentsSection<G1AffinePoint>;

struct CoefficientsSection {
  std::vector<std::vector<Cell>> a;
  std::vector<std::vector<Cell>> b;

  bool operator==(const CoefficientsSection& other) const {
    return a == other.a && b == other.b;
  }
  bool operator!=(const CoefficientsSection& other) const {
    return !operator==(other);
  }

  bool Read(const base::ReadOnlyBuffer& buffer, uint32_t domain_size,
            uint32_t field_size) {
    base::EndianAutoReset reset(buffer, base::Endian::kLittle);
    uint32_t num_coefficients;
    if (!buffer.Read(&num_coefficients)) return false;
    a.resize(domain_size);
    b.resize(domain_size);
    for (uint32_t i = 0; i < num_coefficients; ++i) {
      Cell cell;
      uint32_t matrix, constraint;
      if (!buffer.ReadMany(&matrix, &constraint, &cell.signal)) return false;
      if (!cell.coefficient.Read(buffer, field_size)) return false;
      if (matrix == 0) {
        a[constraint].push_back(std::move(cell));
      } else {
        b[constraint].push_back(std::move(cell));
      }
    }
    return true;
  }

  template <typename F>
  void Normalize() {
    for (std::vector<Cell>& constraint : a) {
      for (Cell& cell : constraint) {
        cell.coefficient.Normalize<F>();
      }
    }
  }

  std::string ToString() const {
    return absl::Substitute("{a: $0, b: $1}", base::Vector2DToString(a),
                            base::Vector2DToString(b));
  }
};

struct ZKey : public circom::ZKey {
  ZKeyHeaderSection header;
  ZKeyHeaderGrothSection header_groth;
  ICSection ic;
  CoefficientsSection coefficients;
  PointsA1Section points_a1;
  PointsB1Section points_b1;
  PointsB2Section points_b2;
  PointsC1Section points_c1;
  PointsH1Section points_h1;

  // circom::ZKey methods
  uint32_t GetVersion() const override { return 1; }
  ZKey* ToV1() override { return this; }

  bool Read(const base::ReadOnlyBuffer& buffer) override {
    Sections<ZKeySectionType> sections(buffer, &ZKeySectionTypeToString);
    if (!sections.Read()) return false;

    if (!sections.MoveTo(ZKeySectionType::kHeader)) return false;
    if (!header.Read(buffer)) return false;

    if (!sections.MoveTo(ZKeySectionType::kHeaderGroth)) return false;
    if (!header_groth.Read(buffer)) return false;
    uint32_t q_field_size = header_groth.q.bytes.size();
    uint32_t r_field_size = header_groth.r.bytes.size();
    uint32_t num_vars = header_groth.num_vars;
    uint32_t num_public_inputs = header_groth.num_public_inputs;
    uint32_t domain_size = header_groth.domain_size;

    if (!sections.MoveTo(ZKeySectionType::kIC)) return false;
    if (!ic.Read(buffer, num_public_inputs + 1, q_field_size)) return false;

    if (!sections.MoveTo(ZKeySectionType::kCoefficients)) return false;
    if (!coefficients.Read(buffer, domain_size, r_field_size)) return false;

    if (!sections.MoveTo(ZKeySectionType::kPointsA1)) return false;
    if (!points_a1.Read(buffer, num_vars, q_field_size)) return false;

    if (!sections.MoveTo(ZKeySectionType::kPointsB1)) return false;
    if (!points_b1.Read(buffer, num_vars, q_field_size)) return false;

    if (!sections.MoveTo(ZKeySectionType::kPointsB2)) return false;
    if (!points_b2.Read(buffer, num_vars, q_field_size)) return false;

    if (!sections.MoveTo(ZKeySectionType::kPointsC1)) return false;
    if (!points_c1.Read(buffer, num_vars, q_field_size)) return false;

    if (!sections.MoveTo(ZKeySectionType::kPointsH1)) return false;
    if (!points_h1.Read(buffer, num_vars, q_field_size)) return false;
    return true;
  }

  template <typename Fq, typename Fr>
  void Normalize() {
    ic.Normalize<Fq>();
    coefficients.Normalize<Fr>();
    points_a1.Normalize<Fq>();
    points_b1.Normalize<Fq>();
    points_b2.Normalize<Fq>();
    points_c1.Normalize<Fq>();
    points_h1.Normalize<Fq>();
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
