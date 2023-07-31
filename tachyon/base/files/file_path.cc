// Copyright (c) 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/files/file_path.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/build/build_config.h"

namespace tachyon::base {
namespace {

const char* const kCommonDoubleExtensionSuffixes[] = {
    "bz", "bz2", "gz", "lz", "lzma", "lzo", "xz", "z", "zst"};

bool AreAllSeparators(const std::string& input) {
  for (char it : input) {
    if (!FilePath::IsSeparator(it)) return false;
  }

  return true;
}

// Find the position of the '.' that separates the extension from the rest
// of the file name. The position is relative to BaseName(), not value().
// Returns npos if it can't find an extension.
std::string::size_type FinalExtensionSeparatorPosition(const std::string& path) {
  // Special case "." and ".."
  if (path == FilePath::kCurrentDirectory || path == FilePath::kParentDirectory)
    return std::string::npos;

  return path.rfind(FilePath::kExtensionSeparator);
}

// Same as above, but allow a second extension component of up to 4
// characters when the rightmost extension component is a common double
// extension (gz, bz2, Z).  For example, foo.tar.gz or foo.tar.Z would have
// extension components of '.tar.gz' and '.tar.Z' respectively.
std::string::size_type ExtensionSeparatorPosition(const std::string& path) {
  const std::string::size_type last_dot = FinalExtensionSeparatorPosition(path);

  // No extension, or the extension is the whole filename.
  if (last_dot == std::string::npos || last_dot == 0U)
    return last_dot;

  const std::string::size_type penultimate_dot =
      path.rfind(FilePath::kExtensionSeparator, last_dot - 1);
  const std::string::size_type last_separator =
      path.find_last_of(FilePath::kSeparators, last_dot - 1,
                        FilePath::kSeparatorsLength - 1);

  if (penultimate_dot == std::string::npos ||
      (last_separator != std::string::npos &&
       penultimate_dot < last_separator)) {
    return last_dot;
  }

  std::string extension(path, last_dot + 1);
  for (auto* i : kCommonDoubleExtensionSuffixes) {
    if (EqualsCaseInsensitiveASCII(extension, i)) {
      if ((last_dot - penultimate_dot) <= 5U &&
          (last_dot - penultimate_dot) > 1U) {
        return penultimate_dot;
      }
    }
  }

  return last_dot;
}

// Returns true if path is "", ".", or "..".
bool IsEmptyOrSpecialCase(std::string_view path) {
  // Special cases "", ".", and ".."
  if (path.empty() || path == FilePath::kCurrentDirectory ||
      path == FilePath::kParentDirectory) {
    return true;
  }

  return false;
}

}  // namespace

FilePath::FilePath() = default;

FilePath::FilePath(std::string_view path) : path_(std::string(path)) {}

FilePath::FilePath(const FilePath& other) = default;
FilePath& FilePath::operator=(const FilePath& other) = default;

FilePath::FilePath(FilePath&& other) noexcept = default;
FilePath& FilePath::operator=(FilePath&& other) noexcept = default;

FilePath::~FilePath() = default;

bool FilePath::IsRoot() const {
  return path_.length() > 0 && AreAllSeparators(path_);
}

// static
bool FilePath::IsSeparator(char character) {
  for (size_t i = 0; i < kSeparatorsLength - 1; ++i) {
    if (character == kSeparators[i]) {
      return true;
    }
  }

  return false;
}

std::vector<std::string> FilePath::GetComponents() const {
  if (value().empty()) return {};

  std::vector<std::string> ret_val;
  FilePath current = *this;
  FilePath base;

  // Capture path components.
  while (current != current.DirName()) {
    base = current.BaseName();
    if (!AreAllSeparators(base.value())) ret_val.push_back(base.value());
    current = current.DirName();
  }

  // Capture root, if any.
  if (current.IsRoot()) ret_val.push_back(FilePath::kRootPath);

  return {ret_val.rbegin(), ret_val.rend()};
}

bool FilePath::IsParent(const FilePath& child) const {
  return AppendRelativePath(child, nullptr);
}

bool FilePath::AppendRelativePath(const FilePath& child, FilePath* path) const {
  std::vector<std::string> parent_components = GetComponents();
  std::vector<std::string> child_components = child.GetComponents();

  if (parent_components.empty() ||
      parent_components.size() >= child_components.size())
    return false;

  std::vector<std::string>::const_iterator parent_comp =
      parent_components.begin();
  std::vector<std::string>::const_iterator child_comp =
      child_components.begin();

  while (parent_comp != parent_components.end()) {
    if (*parent_comp != *child_comp) return false;
    ++parent_comp;
    ++child_comp;
  }

  if (path) {
    *path = path->Append(FilePath(
        absl::StrJoin(child_comp, child_components.cend(), kSeparators)));
  }
  return true;
}

FilePath FilePath::DirName() const {
  std::string_view p = path_;
  size_t last_separator = p.find_last_of(kSeparators, std::string::npos);
  if (last_separator == std::string::npos) {
    return FilePath();
  } else if (last_separator == 0) {
    // root directory
    return FilePath(kRootPath);
  } else {
    return FilePath(p.substr(0, last_separator));
  }
}

FilePath FilePath::BaseName() const {
  std::string_view p = path_;
  size_t last_separator = p.find_last_of(kSeparators, std::string::npos);
  if (last_separator == std::string::npos) {
    return FilePath(path_);
  }

  return FilePath(
      p.substr(last_separator + 1, p.length() - last_separator - 1));
}

std::string FilePath::Extension() const {
  FilePath base(BaseName());
  std::string_view p = base.value();
  size_t last_separator =
      p.find_last_of(kExtensionSeparator, std::string::npos);
  if (last_separator == std::string::npos) return EmptyString();
  return std::string(p.substr(last_separator, p.length() - last_separator));
}

std::string FilePath::FinalExtension() const {
  FilePath base(BaseName());
  const std::string::size_type dot = FinalExtensionSeparatorPosition(base.path_);
  if (dot == std::string::npos)
    return std::string();

  return base.path_.substr(dot, std::string::npos);
}

FilePath FilePath::RemoveExtension() const {
  if (Extension().empty())
    return *this;

  const std::string::size_type dot = ExtensionSeparatorPosition(path_);
  if (dot == std::string::npos)
    return *this;

  return FilePath(path_.substr(0, dot));
}

FilePath FilePath::RemoveFinalExtension() const {
  if (FinalExtension().empty())
    return *this;

  const std::string::size_type dot = FinalExtensionSeparatorPosition(path_);
  if (dot == std::string::npos)
    return *this;

  return FilePath(path_.substr(0, dot));
}

FilePath FilePath::InsertBeforeExtension(std::string_view suffix) const {
  if (suffix.empty())
    return FilePath(path_);

  if (IsEmptyOrSpecialCase(BaseName().value()))
    return FilePath();

  return FilePath(
      absl::StrCat(RemoveExtension().value(), suffix, Extension()));
}

FilePath FilePath::Append(std::string_view component) const {
  return Append(FilePath(component));
}

FilePath FilePath::Append(const FilePath& component) const {
  if (component.IsAbsolute()) return FilePath(EmptyString());
  if (path_.empty()) return component;
  if (EndsWithSeparator()) {
    return FilePath(absl::StrCat(path_, component.value()));
  }
  return FilePath(absl::StrJoin({path_, component.value()}, kSeparators));
}

FilePath FilePath::operator/(std::string_view component) const {
  return Append(component);
}

FilePath FilePath::operator/(const FilePath& component) const {
  return Append(component);
}

FilePath& FilePath::operator/=(std::string_view component) {
  *this = Append(component);
  return *this;
}

FilePath& FilePath::operator/=(const FilePath& component) {
  *this = Append(component);
  return *this;
}

bool FilePath::IsAbsolute() const {
 return path_.length() > 0 && FilePath::IsSeparator(path_[0]);
}

bool FilePath::EndsWithSeparator() const {
  if (path_.empty()) return false;
  return IsSeparator(path_.back());
}

FilePath FilePath::AsEndingWithSeparator() const {
  if (EndsWithSeparator() || path_.empty()) return *this;

  std::string path_str;
  path_str.reserve(path_.length() + 1);  // Only allocate string once.
  path_str = path_;
  path_str.push_back(kSeparators[0]);
  return FilePath(path_str);
}

FilePath FilePath::StripTrailingSeparators() const {
  FilePath new_path(path_);
  new_path.StripTrailingSeparatorsInternal();

  return new_path;
}

void FilePath::StripTrailingSeparatorsInternal() {
  size_t pos = path_.find_last_not_of(kSeparators);
  if (pos == std::string::npos) {
    path_.resize(1);
  } else {
    path_.resize(pos + 1);
  }
}

bool FilePath::ReferencesParent() const {
  return path_.find(kParentDirectory) != std::string::npos;
}

std::ostream& operator<<(std::ostream& out, const FilePath& file_path) {
  out << file_path.value();
  return out;
}

}  // namespace tachyon::base
