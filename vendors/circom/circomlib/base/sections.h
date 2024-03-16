#ifndef VENDORS_CIRCOM_CIRCOMLIB_BASE_SECTIONS_H_
#define VENDORS_CIRCOM_CIRCOMLIB_BASE_SECTIONS_H_

#include <stdint.h>

#include <string_view>
#include <vector>

#include "tachyon/base/buffer/endian_auto_reset.h"
#include "tachyon/base/buffer/read_only_buffer.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/range.h"

namespace tachyon::circom {

template <typename T>
struct Section {
  T type;
  base::Range<uint64_t> range;
};

template <typename T>
class Sections {
 public:
  typedef std::string_view (*ErrorFn)(T type);

  Sections(const base::ReadOnlyBuffer& buffer, ErrorFn error_fn)
      : buffer_(buffer), error_fn_(error_fn) {}

  bool Read() {
    base::EndianAutoReset reset(buffer_, base::Endian::kLittle);
    uint32_t num_sections;
    if (!buffer_.Read(&num_sections)) return false;

    sections_.reserve(num_sections);
    for (uint32_t i = 0; i < num_sections; ++i) {
      if (!Add()) return false;
    }
    return true;
  }

  bool MoveTo(T type) const {
    auto it = std::find_if(
        sections_.begin(), sections_.end(),
        [type](const Section<T>& section) { return section.type == type; });
    if (it == sections_.end()) {
      LOG(ERROR) << error_fn_(type) << " is empty";
      return false;
    }
    buffer_.set_buffer_offset(it->range.from);
    return true;
  }

 private:
  bool Add() {
    T type;
    uint64_t size;
    if (!buffer_.ReadMany(&type, &size)) return false;

    sections_.push_back({type, {buffer_.buffer_offset(), size}});
    buffer_.set_buffer_offset(buffer_.buffer_offset() + size);
    return true;
  }

  const base::ReadOnlyBuffer& buffer_;
  ErrorFn error_fn_;
  std::vector<Section<T>> sections_;
};

}  // namespace tachyon::circom

#endif  // VENDORS_CIRCOM_CIRCOMLIB_BASE_SECTIONS_H_
