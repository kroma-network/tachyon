// Copyright (c) 2020 The Console Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/flag/flag.h"

#include <algorithm>

namespace tachyon {
namespace base {
namespace {

size_t Append(std::ostream& ss, std::string_view text) {
  ss << text;
  return text.length();
}

std::string AlignAtIndexAndAppend(std::stringstream& ss, std::string_view text,
                                  int index, int index2) {
  int final_index = index;
  if (final_index <= 0) {
    ss << std::endl;
    final_index = index2;
  }
  ss << std::string(final_index, ' ') << text;
  return ss.str();
}

}  // namespace

bool IsValidFlagName(std::string_view text) {
  if (text.length() == 0) return false;
  if (!absl::ascii_isalpha(text[0])) return false;
  text.remove_prefix(1);
  return std::all_of(text.begin(), text.end(), [](char c) {
    return absl::ascii_isalpha(c) || absl::ascii_isdigit(c) || c == '_';
  });
}

FlagBase::FlagBase() = default;

FlagBase::~FlagBase() = default;

const std::string& FlagBase::display_name() const {
  if (!name_.empty()) return name_;
  if (!long_name_.empty()) return long_name_;
  return short_name_;
}

std::string FlagBase::display_help(int help_start) const {
  int remain_len = help_start;
  std::stringstream ss;
  if (is_positional()) {
    remain_len -= Append(ss, name_);
  } else {
    if (!short_name_.empty()) {
      remain_len -= Append(ss, short_name_);
    }

    if (!long_name_.empty()) {
      if (!short_name_.empty()) {
        remain_len -= Append(ss, ", ");
      }
      remain_len -= Append(ss, long_name_);
    }
  }

  return AlignAtIndexAndAppend(ss, help_, remain_len, help_start);
}

bool FlagBase::ConsumeNamePrefix(FlagParserBase& parser,
                                 std::string_view* arg) const {
  std::string_view input = *arg;
  if (!long_name_.empty()) {
    if (ConsumePrefix(&input, long_name_)) {
      if (input.empty() || input[0] == '=') {
        *arg = input;
        return true;
      }
    }
  }
  if (!short_name_.empty()) {
    if (ConsumePrefix(&input, short_name_)) {
      if (input.empty() || input[0] == '=') {
        *arg = input;
        return true;
      }
    }
  }
  return false;
}

}  // namespace base
}  // namespace tachyon
