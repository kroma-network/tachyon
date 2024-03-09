#ifndef VENDORS_CIRCOM_CIRCOMLIB_R1CS_R1CS_H_
#define VENDORS_CIRCOM_CIRCOMLIB_R1CS_R1CS_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "tachyon/base/buffer/copyable.h"
#include "tachyon/base/buffer/endian_auto_reset.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/math/base/big_int.h"

namespace tachyon {
namespace circom {
namespace v1 {

struct R1CS;

}  // namespace v1

struct R1CS {
  virtual ~R1CS() = default;

  virtual uint32_t GetVersion() const = 0;

  virtual v1::R1CS* ToV1() { return nullptr; }
};

constexpr char kR1CSMagic[4] = {'r', '1', 'c', 's'};

struct PrimeField {
  std::vector<uint8_t> bytes;

  bool operator==(const PrimeField& other) const {
    return bytes == other.bytes;
  }
  bool operator!=(const PrimeField& other) const {
    return bytes != other.bytes;
  }

  template <size_t N>
  math::BigInt<N> ToBigInt() const {
    return math::BigInt<N>::FromBytesLE(bytes);
  }

  template <size_t N>
  static PrimeField FromBigInt(const math::BigInt<N>& big_int) {
    std::array<uint8_t, N * 8> bytes = big_int.ToBytesLE();
    return {{bytes.begin(), bytes.end()}};
  }

  std::string ToString() const;
};

namespace v1 {

enum class R1CSSectionType : uint32_t {
  kHeader = 0x1,
  kConstraints = 0x2,
  kWire2LabelIdMap = 0x3,
  kCustomGatesList = 0x4,
  kCustomGatesApplication = 0x5,
};

std::string_view R1CSSectionTypeToString(R1CSSectionType type);

struct R1CSHeaderSection {
  PrimeField modulus;
  // Total number of wires including ONE signal (Index 0).
  uint32_t num_wires;
  // Total number of public output wires. They should be starting at
  // idx 1.
  uint32_t num_public_outputs;
  // Total number of public input wires. They should be starting just
  // after the public output.
  uint32_t num_public_inputs;
  // Total number of private input wires. They should be starting just
  // after the public inputs.
  uint32_t num_private_inputs;
  // Total number of labels.
  uint64_t num_labels;
  // Total number of constraints.
  uint32_t num_constraints;

  bool operator==(const R1CSHeaderSection& other) const {
    return modulus == other.modulus && num_wires == other.num_wires &&
           num_public_outputs == other.num_public_outputs &&
           num_public_inputs == other.num_public_inputs &&
           num_private_inputs == other.num_private_inputs &&
           num_labels == other.num_labels &&
           num_constraints == other.num_constraints;
  }
  bool operator!=(const R1CSHeaderSection& other) const {
    return !operator==(other);
  }

  std::string ToString() const {
    return absl::Substitute(
        "{modulus: $0, num_wires: $1, num_public_outputs: $2, "
        "num_public_inputs: $3, num_private_inputs: $4 num_labels: $5, "
        "num_constraints: $6}",
        modulus.ToString(), num_wires, num_public_outputs, num_public_inputs,
        num_private_inputs, num_labels, num_constraints);
  }
};

struct Term {
  uint32_t wire_id;
  PrimeField coefficient;

  bool operator==(const Term& other) const {
    return wire_id == other.wire_id && coefficient == other.coefficient;
  }
  bool operator!=(const Term& other) const { return !operator==(other); }

  std::string ToString() const {
    return absl::Substitute("$0ω_$1", coefficient.ToString(), wire_id);
  }
};

struct LinearCombination {
  std::vector<Term> terms;

  bool operator==(const LinearCombination& other) const {
    return terms == other.terms;
  }
  bool operator!=(const LinearCombination& other) const {
    return terms != other.terms;
  }

  std::string ToString() const {
    std::stringstream ss;
    for (size_t i = 0; i < terms.size(); ++i) {
      ss << terms[i].ToString();
      if (i != terms.size() - 1) ss << " + ";
    }
    return ss.str();
  }
};

struct Constraint {
  LinearCombination a;
  LinearCombination b;
  LinearCombination c;

  bool operator==(const Constraint& other) const {
    return a == other.a && b == other.b && c == other.c;
  }
  bool operator!=(const Constraint& other) const { return !operator==(other); }

  std::string ToString() const {
    if (a.terms.size() > 1 && b.terms.size() > 1) {
      return absl::Substitute("($0) * ($1) = $2", a.ToString(), b.ToString(),
                              c.ToString());
    } else if (a.terms.size() > 1) {
      return absl::Substitute("($0) * $1 = $2", a.ToString(), b.ToString(),
                              c.ToString());
    } else if (b.terms.size() > 1) {
      return absl::Substitute("$0 * ($1) = $2", a.ToString(), b.ToString(),
                              c.ToString());
    } else {
      return absl::Substitute("$0 * $1 = $2", a.ToString(), b.ToString(),
                              c.ToString());
    }
  }
};

struct R1CSConstraintsSection {
  std::vector<Constraint> constraints;

  bool operator==(const R1CSConstraintsSection& other) const {
    return constraints == other.constraints;
  }
  bool operator!=(const R1CSConstraintsSection& other) const {
    return constraints != other.constraints;
  }

  std::string ToString() const { return base::VectorToString(constraints); }
};

struct R1CSWireId2LabelIdMapSection {
  std::vector<uint64_t> label_ids;

  bool operator==(const R1CSWireId2LabelIdMapSection& other) const {
    return label_ids == other.label_ids;
  }
  bool operator!=(const R1CSWireId2LabelIdMapSection& other) const {
    return label_ids != other.label_ids;
  }

  std::string ToString() const { return base::VectorToString(label_ids); }
};

struct R1CS : public circom::R1CS {
  R1CSHeaderSection header;
  R1CSConstraintsSection constraints;
  R1CSWireId2LabelIdMapSection wire_id_to_label_id_map;

  // circom::R1CS methods
  uint32_t GetVersion() const override { return 1; }
  v1::R1CS* ToV1() override { return this; }

  std::string ToString() const {
    return absl::Substitute(
        "{header: $0, constraints: $1, wire_id_to_label_id_map: $2}",
        header.ToString(), constraints.ToString(),
        wire_id_to_label_id_map.ToString());
  }
};

}  // namespace v1
}  // namespace circom

namespace base {

template <>
class Copyable<circom::v1::R1CSHeaderSection> {
 public:
  static bool WriteTo(const circom::v1::R1CSHeaderSection& header,
                      Buffer* buffer) {
    base::EndianAutoReset reset(*buffer, base::Endian::kLittle);
    uint32_t field_size = header.modulus.bytes.size();
    if (!buffer->Write(field_size)) return false;
    if (!buffer->Write(header.modulus.bytes.data(), field_size)) return false;
    return buffer->WriteMany(
        header.num_wires, header.num_public_outputs, header.num_public_inputs,
        header.num_private_inputs, header.num_labels, header.num_constraints);
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer,
                       circom::v1::R1CSHeaderSection* header) {
    base::EndianAutoReset reset(buffer, base::Endian::kLittle);
    uint32_t field_size;
    if (!buffer.Read(&field_size)) return false;
    if (field_size % 8 != 0) {
      LOG(ERROR) << "field size is not a multiple of 8";
      return false;
    }
    std::vector<uint8_t> bytes(field_size);
    if (!buffer.Read(bytes.data(), bytes.size())) return false;
    uint32_t num_wires;
    uint32_t num_public_outputs;
    uint32_t num_public_inputs;
    uint32_t num_private_inputs;
    uint64_t num_labels;
    uint32_t num_constraints;
    if (!buffer.ReadMany(&num_wires, &num_public_outputs, &num_public_inputs,
                         &num_private_inputs, &num_labels, &num_constraints))
      return false;
    *header = {
        {bytes},           num_wires,          num_public_outputs,
        num_public_inputs, num_private_inputs, num_labels,
        num_constraints,
    };
    return true;
  }

  static size_t EstimateSize(const circom::v1::R1CSHeaderSection& header) {
    uint32_t field_size = header.modulus.bytes.size();
    return sizeof(uint32_t) + field_size * 8 +
           base::EstimateSize(header.num_wires, header.num_public_outputs,
                              header.num_public_inputs,
                              header.num_private_inputs, header.num_labels,
                              header.num_constraints);
  }
};

template <>
class Copyable<circom::v1::R1CS> {
 public:
  static bool WriteTo(const circom::v1::R1CS& r1cs, Buffer* buffer) {
    NOTIMPLEMENTED();
    return false;
  }

  static bool ReadFrom(const ReadOnlyBuffer& buffer, circom::v1::R1CS* r1cs) {
    base::EndianAutoReset reset(buffer, base::Endian::kLittle);
    uint32_t num_sections;
    if (!buffer.Read(&num_sections)) return false;
    circom::v1::R1CSHeaderSection header;
    size_t constraints_section_offset;
    size_t wire_to_labe_id_map_offset;
    for (uint32_t i = 0; i < num_sections; ++i) {
      circom::v1::R1CSSectionType section_type;
      uint64_t section_size;
      if (!buffer.ReadMany(&section_type, &section_size)) {
        return false;
      }
      switch (section_type) {
        case circom::v1::R1CSSectionType::kHeader: {
          if (!buffer.Read(&header)) return false;
          break;
        }
        case circom::v1::R1CSSectionType::kConstraints: {
          constraints_section_offset = buffer.buffer_offset();
          buffer.set_buffer_offset(buffer.buffer_offset() + section_size);
          break;
        }
        case circom::v1::R1CSSectionType::kWire2LabelIdMap: {
          wire_to_labe_id_map_offset = buffer.buffer_offset();
          buffer.set_buffer_offset(buffer.buffer_offset() + section_size);
          break;
        }
        case circom::v1::R1CSSectionType::kCustomGatesList:
        case circom::v1::R1CSSectionType::kCustomGatesApplication: {
          NOTIMPLEMENTED();
          return false;
        }
      }
    }
    buffer.set_buffer_offset(constraints_section_offset);
    uint32_t field_size = header.modulus.bytes.size();
    circom::v1::R1CSConstraintsSection constraints;
    for (uint32_t i = 0; i < header.num_constraints; ++i) {
      circom::v1::Constraint constraint;
      for (uint32_t j = 0; j < 3; ++j) {
        uint32_t n;
        if (!buffer.Read(&n)) return false;
        std::vector<circom::v1::Term> terms(n);
        for (uint32_t k = 0; k < n; ++k) {
          if (!buffer.Read(&terms[k].wire_id)) return false;
          terms[k].coefficient.bytes.resize(field_size);
          if (!buffer.Read(terms[k].coefficient.bytes.data(), field_size))
            return false;
        }
        if (j == 0) {
          constraint.a = {std::move(terms)};
        } else if (j == 1) {
          constraint.b = {std::move(terms)};
        } else {
          constraint.c = {std::move(terms)};
        }
      }
      constraints.constraints.push_back(std::move(constraint));
    }

    buffer.set_buffer_offset(wire_to_labe_id_map_offset);
    circom::v1::R1CSWireId2LabelIdMapSection wire_id_to_label_id_map;
    wire_id_to_label_id_map.label_ids.resize(header.num_wires);
    for (uint32_t i = 0; i < header.num_wires; ++i) {
      if (!buffer.Read(&wire_id_to_label_id_map.label_ids[i])) return false;
    }

    r1cs->header = std::move(header);
    r1cs->constraints = std::move(constraints);
    r1cs->wire_id_to_label_id_map = std::move(wire_id_to_label_id_map);
    return true;
  }

  static size_t EstimateSize(const circom::v1::R1CS& r1cs) {
    NOTIMPLEMENTED();
    return 0;
  }
};

}  // namespace base
}  // namespace tachyon

#endif  // VENDORS_CIRCOM_CIRCOMLIB_R1CS_R1CS_H_
