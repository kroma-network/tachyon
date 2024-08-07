#ifndef VENDORS_CIRCOM_CIRCOMLIB_WTNS_WTNS_H_
#define VENDORS_CIRCOM_CIRCOMLIB_WTNS_WTNS_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>

#include <memory>
#include <string>
#include <utility>

#include "circomlib/base/modulus.h"
#include "circomlib/base/sections.h"
#include "tachyon/base/buffer/endian_auto_reset.h"
#include "tachyon/base/buffer/read_only_buffer.h"
#include "tachyon/base/files/bin_file.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/openmp_util.h"
#include "tachyon/base/strings/string_util.h"

namespace tachyon::circom {
namespace v2 {

template <typename F>
struct Wtns;

}  // namespace v2

template <typename F>
struct Wtns {
  explicit Wtns(base::BinFile bin_file) : bin_file(std::move(bin_file)) {}
  virtual ~Wtns() = default;

  virtual uint32_t GetVersion() const = 0;

  virtual v2::Wtns<F>* ToV2() { return nullptr; }

  virtual bool Read(const base::ReadOnlyBuffer& buffer) = 0;

  virtual size_t GetNumWitness() const = 0;

  virtual absl::Span<const F> GetWitnesses() const = 0;

  base::BinFile bin_file;
};

constexpr char kWtnsMagic[4] = {'w', 't', 'n', 's'};

// Return nullptr if the parser failed to parse.
template <typename F>
std::unique_ptr<Wtns<F>> ParseWtns(const base::FilePath& path,
                                   bool use_mmap = true) {
  base::BinFile bin_file;
  CHECK(bin_file.Load(path, use_mmap));
  if (use_mmap) {
    PCHECK(madvise(bin_file.GetData(), bin_file.GetDataLength(),
                   MADV_SEQUENTIAL) == 0);
  }

  base::ReadOnlyBuffer buffer = bin_file.ToReadOnlyBuffer();
  buffer.set_endian(base::Endian::kLittle);
  char magic[4];
  uint32_t version;
  if (!buffer.ReadMany(magic, &version)) return nullptr;
  if (memcmp(magic, kWtnsMagic, 4) != 0) {
    LOG(ERROR) << "Invalid magic: " << magic;
    return nullptr;
  }
  std::unique_ptr<Wtns<F>> wtns;
  if (version == 2) {
    wtns.reset(new v2::Wtns<F>(std::move(bin_file)));
    CHECK(wtns->ToV2()->Read(buffer));
  } else {
    LOG(ERROR) << "Invalid version: " << version;
    return nullptr;
  }
  return wtns;
}

namespace v2 {

enum class WtnsSectionType : uint32_t {
  kHeader = 0x1,
  kData = 0x2,
};

std::string_view WtnsSectionTypeToString(WtnsSectionType type);

struct WtnsHeaderSection {
  Modulus modulus;
  uint32_t num_witness;

  bool operator==(const WtnsHeaderSection& other) const {
    return modulus == other.modulus && num_witness == other.num_witness;
  }
  bool operator!=(const WtnsHeaderSection& other) const {
    return !operator==(other);
  }

  bool Read(const base::ReadOnlyBuffer& buffer) {
    base::EndianAutoReset reset(buffer, base::Endian::kLittle);
    if (!modulus.Read(buffer)) return false;
    return buffer.ReadMany(&num_witness);
  }

  std::string ToString() const {
    return absl::Substitute("{modulus: $0, num_witness: $1}",
                            modulus.ToString(), num_witness);
  }
};

template <typename F>
struct WtnsDataSection {
  absl::Span<F> witnesses;

  bool operator==(const WtnsDataSection& other) const {
    return witnesses == other.witnesses;
  }
  bool operator!=(const WtnsDataSection& other) const {
    return witnesses != other.witnesses;
  }

  bool Read(const base::ReadOnlyBuffer& buffer,
            const WtnsHeaderSection& header) {
    F* ptr;
    if (!buffer.ReadPtr(&ptr, header.num_witness)) return false;
    witnesses = {ptr, header.num_witness};

    OMP_PARALLEL_FOR(uint32_t i = 0; i < header.num_witness; ++i) {
      witnesses[i] = F(witnesses[i].value());
    }
    return true;
  }

  std::string ToString() const { return base::ContainerToString(witnesses); }
};

template <typename F>
struct Wtns : public circom::Wtns<F> {
  WtnsHeaderSection header;
  WtnsDataSection<F> data;

  explicit Wtns(base::BinFile bin_file)
      : circom::Wtns<F>(std::move(bin_file)) {}

  // circom::Wtns methods
  uint32_t GetVersion() const override { return 2; }
  Wtns* ToV2() override { return this; }

  bool Read(const base::ReadOnlyBuffer& buffer) override {
    Sections<WtnsSectionType> sections(buffer, &WtnsSectionTypeToString);
    if (!sections.Read()) return false;

    if (!sections.MoveTo(WtnsSectionType::kHeader)) return false;
    if (!header.Read(buffer)) return false;

    if (!sections.MoveTo(WtnsSectionType::kData)) return false;
    if (!data.Read(buffer, header)) return false;
    return true;
  }

  size_t GetNumWitness() const override { return data.witnesses.size(); }

  absl::Span<const F> GetWitnesses() const override { return data.witnesses; }

  std::string ToString() const {
    return absl::Substitute("{header: $0, data: $1}", header.ToString(),
                            data.ToString());
  }
};

}  // namespace v2
}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_WTNS_WTNS_H_
