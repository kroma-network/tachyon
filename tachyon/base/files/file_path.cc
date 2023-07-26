// Copyright (c) 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/files/file_path.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/build/build_config.h"

namespace tachyon {
namespace base {
namespace {

#if BUILDFLAG(IS_POSIX)
constexpr const char* kRootPath = "/";
constexpr const char* kSeparator = "/";
constexpr const char* kExtensionSeparator = ".";
constexpr const char* kParentDirectory = "..";
#else
#error Unsupported platform
#endif

bool AreAllSeparators(const std::string& input) {
  for (char it : input) {
    if (it != kSeparator[0]) return false;
  }

  return true;
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

void FilePath::GetComponents(std::vector<std::string>* components) const {
  DCHECK(components);
  if (!components) return;
  components->clear();
  if (value().empty()) return;

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
  if (current.IsRoot()) ret_val.push_back(kRootPath);

  *components = std::vector<std::string>(ret_val.rbegin(), ret_val.rend());
}

bool FilePath::IsParent(const FilePath& child) const {
  return AppendRelativePath(child, nullptr);
}

bool FilePath::AppendRelativePath(const FilePath& child, FilePath* path) const {
  std::vector<std::string> parent_components;
  std::vector<std::string> child_components;
  GetComponents(&parent_components);
  child.GetComponents(&child_components);

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
        absl::StrJoin(child_comp, child_components.cend(), kSeparator)));
  }
  return true;
}

FilePath FilePath::DirName() const {
  std::string_view p = path_;
  size_t last_separator = p.find_last_of(kSeparator, std::string::npos);
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
  size_t last_separator = p.find_last_of(kSeparator, std::string::npos);
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

FilePath FilePath::Append(std::string_view component) const {
  return Append(FilePath(component));
}

FilePath FilePath::Append(const FilePath& component) const {
  if (component.IsAbsolute()) return FilePath(EmptyString());
  if (path_.empty()) return component;
  if (EndsWithSeparator()) {
    return FilePath(absl::StrCat(path_, component.value()));
  }
  return FilePath(absl::StrJoin({path_, component.value()}, kSeparator));
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

bool FilePath::IsAbsolute() const { return StartsWith(path_, kSeparator); }

bool FilePath::EndsWithSeparator() const {
  if (path_.empty()) return false;
  return EndsWith(path_, kSeparator);
}

FilePath FilePath::AsEndingWithSeparator() const {
  if (EndsWithSeparator() || path_.empty()) return *this;

  std::string path_str;
  path_str.reserve(path_.length() + 1);  // Only allocate string once.
  path_str = path_;
  path_str.push_back(kSeparator[0]);
  return FilePath(path_str);
}

FilePath FilePath::StripTrailingSeparators() const {
  FilePath new_path(path_);
  new_path.StripTrailingSeparatorsInternal();

  return new_path;
}

void FilePath::StripTrailingSeparatorsInternal() {
  size_t pos = path_.find_last_not_of(kSeparator);
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

}  // namespace base
}  // namespace tachyon
