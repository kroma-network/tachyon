// Copyright (c) 2020 The Console Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_FLAG_FLAG_H_
#define TACHYON_BASE_FLAG_FLAG_H_

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "gtest/gtest_prod.h"

#include "tachyon/base/containers/contains.h"
#include "tachyon/base/environment.h"
#include "tachyon/base/flag/flag_forward.h"
#include "tachyon/base/flag/flag_value_traits.h"
#include "tachyon/base/logging.h"
#include "tachyon/base/strings/string_util.h"
#include "tachyon/export.h"

namespace tachyon::base {

TACHYON_EXPORT bool IsValidFlagName(std::string_view text);

class FlagParserBase;
class FlagParser;
class SubParser;

template <typename T>
class FlagBaseBuilder {
 public:
  T& set_short_name(std::string_view short_name) {
    T* impl = static_cast<T*>(this);
    std::string_view text = short_name;
    if (!ConsumePrefix(&text, "-")) return *impl;
    if (!(text.length() == 1 && absl::ascii_isalpha(text[0]))) return *impl;

    impl->short_name_ = std::string(short_name);
    return *impl;
  }

  T& set_long_name(std::string_view long_name) {
    T* impl = static_cast<T*>(this);
    std::string_view text = long_name;
    if (!ConsumePrefix(&text, "--")) return *impl;
    if (!IsValidFlagName(text)) return *impl;

    impl->long_name_ = std::string(long_name);
    return *impl;
  }

  T& set_name(std::string_view name) {
    T* impl = static_cast<T*>(this);
    std::string_view text = name;
    if (!IsValidFlagName(text)) return *impl;

    impl->name_ = std::string(name);
    return *impl;
  }

  T& set_help(std::string_view help) {
    T* impl = static_cast<T*>(this);
    impl->help_ = std::string(help);
    return *impl;
  }

  T& set_required() {
    T* impl = static_cast<T*>(this);
    impl->is_required_ = true;
    return *impl;
  }
};

// FlagBase must have |short_name_|, |long_name_| or |name_|.
// |short_name_| should be a alphabet with a prefix "-".
// |long_name_| should contain alphabet, digit or underscore with a prefix "--".
// |name_| should contain alphabet, digit or underscore without any prefix.
// |name_| and |long_name_| should start with alphabet.
// e.g, --3a or --_ab are not allowed.
// |long_name_| and |short_name_| can be set together, but |name_| shouldn't.
class TACHYON_EXPORT FlagBase {
 public:
  FlagBase();
  FlagBase(const FlagBase& other) = delete;
  FlagBase& operator=(const FlagBase& other) = delete;
  virtual ~FlagBase();

  const std::string& short_name() const { return short_name_; }
  const std::string& long_name() const { return long_name_; }
  const std::string& name() const { return name_; }
  const std::string& help() const { return help_; }

  // Returns true if the flag was marked with required.
  bool is_required() const { return is_required_; }
  // Returns true |name_| was set.
  bool is_positional() const { return !name_.empty(); }
  // Returns true |short_name_| or |long_name_| was set.
  bool is_optional() const {
    return !short_name_.empty() || !long_name_.empty();
  }
  // Returns true if a value was set.
  bool is_set() const { return is_set_; }
  // Returns true if a flag is an instance of SubParser
  virtual bool IsSubParser() const { return false; }

  SubParser* ToSubParser() {
    DCHECK(IsSubParser());
    return reinterpret_cast<SubParser*>(this);
  }

  // Returns |name_| if it is positional.
  // Otherwise, it returns |long_name_| if it is not empty.
  // Returns |short_name_| if |long_name_| is empty.
  const std::string& display_name() const;
  std::string display_help(int help_start = 0) const;

 protected:
  template <typename T>
  friend class FlagBaseBuilder;
  friend class FlagParserBase;
  FRIEND_TEST(FlagParserTest, SubParserTest);

  bool ConsumeNamePrefix(FlagParserBase& parser, std::string_view* arg) const;

  // Returns true if underlying type of Flag<T>, in other words, T is bool.
  virtual bool NeedsValue() const = 0;
  virtual bool ParseValue(std::string_view arg, std::string* reason) = 0;
  virtual bool ParseValueFromEnvironment(std::string* reason) { return true; }

  void reset() { is_set_ = false; }

  std::string short_name_;
  std::string long_name_;
  std::string name_;
  std::string help_;
  bool is_required_ = false;
  bool is_set_ = false;
};

template <typename T, typename value_type>
class FlagBuilder : public FlagBaseBuilder<T> {
 public:
  T& set_default_value(const value_type& value) {
    T* impl = static_cast<T*>(this);
    *(impl->value_) = value;
    return *impl;
  }

  T& set_env_name(std::string_view env_name) {
    T* impl = static_cast<T*>(this);
    impl->env_name_ = std::string(env_name);
    return *impl;
  }
};

template <typename T>
class Flag : public FlagBase, public FlagBuilder<Flag<T>, T> {
 public:
  using value_type = T;
  using ParseValueCallback =
      std::function<bool(std::string_view, std::string*)>;

  explicit Flag(T* value) : value_(value) {}
  explicit Flag(ParseValueCallback parse_value_callback)
      : parse_value_callback_(parse_value_callback) {}
  Flag(const Flag& other) = delete;
  Flag& operator=(const Flag& other) = delete;

 private:
  friend class FlagBuilder<Flag<T>, T>;
  FRIEND_TEST(FlagTest, ParseValue);
  FRIEND_TEST(TimeDeltaFlagTest, ParseValue);

  void set_value(const T& value) {
    is_set_ = true;
    *value_ = value;
  }

  const T* value() const { return value_; }

  bool NeedsValue() const override { return !std::is_same<T, bool>::value; }
  bool ParseValue(std::string_view arg, std::string* reason) override;
  bool ParseValueFromEnvironment(std::string* reason) override;

  T* value_ = nullptr;
  ParseValueCallback parse_value_callback_;
  std::string env_name_;
};

template <typename T>
bool Flag<T>::ParseValue(std::string_view arg, std::string* reason) {
  if (parse_value_callback_) {
    bool ret = parse_value_callback_(arg, reason);
    if (ret) {
      is_set_ = true;
    }
    return ret;
  }

  if (FlagValueTraits<T>::ParseValue(arg, value_, reason)) {
    is_set_ = true;
    return true;
  }
  return false;
}

template <typename T>
bool Flag<T>::ParseValueFromEnvironment(std::string* reason) {
  if (!env_name_.empty()) {
    std::string_view value;
    if (Environment::Get(env_name_, &value)) {
      return ParseValue(value, reason);
    }
  }
  return true;
}

template <typename T, typename value_type>
class ChoicesFlagBuilder : public FlagBaseBuilder<T> {
 public:
  T& set_default_value(const value_type& value) {
    T* impl = static_cast<T*>(this);
    DCHECK(impl->Contains(value));
    *(impl->value_) = value;
    return *impl;
  }

  T& set_env_name(std::string_view env_name) {
    T* impl = static_cast<T*>(this);
    impl->env_name_ = std::string(env_name);
    return *impl;
  }
};

template <typename T>
class ChoicesFlag : public FlagBase,
                    public ChoicesFlagBuilder<ChoicesFlag<T>, T> {
 public:
  using value_type = T;

  ChoicesFlag(T* value, const std::vector<T>& choices)
      : value_(value), choices_(choices) {}
  ChoicesFlag(T* value, std::vector<T>&& choices)
      : value_(value), choices_(std::move(choices)) {}
  ChoicesFlag(const ChoicesFlag& other) = delete;
  ChoicesFlag& operator=(const ChoicesFlag& other) = delete;

 private:
  friend class ChoicesFlagBuilder<ChoicesFlag<T>, T>;
  FRIEND_TEST(FlagTest, ParseValue);

  void set_value(const T& value) {
    is_set_ = true;
    *value_ = value;
  }

  const T* value() const { return value_; }

  bool Contains(const T& value) { return base::Contains(choices_, value); }

  bool NeedsValue() const override { return !std::is_same<T, bool>::value; }
  bool ParseValue(std::string_view arg, std::string* reason) override;
  bool ParseValueFromEnvironment(std::string* reason) override;

  T* value_ = nullptr;
  std::vector<T> choices_;
  std::string env_name_;
};

template <typename T>
bool ChoicesFlag<T>::ParseValue(std::string_view arg, std::string* reason) {
  T value;
  if (FlagValueTraits<T>::ParseValue(arg, &value, reason)) {
    if (Contains(value)) {
      *value_ = std::move(value);
      is_set_ = true;
      return true;
    } else {
      *reason = absl::Substitute("$0 is not in choices", arg);
    }
  }
  return false;
}

template <typename T>
bool ChoicesFlag<T>::ParseValueFromEnvironment(std::string* reason) {
  if (!env_name_.empty()) {
    std::string_view value;
    if (Environment::Get(env_name_, &value)) {
      return ParseValue(value, reason);
    }
  }
  return true;
}

template <typename T, typename value_type>
class RangeFlagBuilder : public FlagBaseBuilder<T> {
 public:
  T& set_default_value(const value_type& value) {
    T* impl = static_cast<T*>(this);
    DCHECK(impl->Contains(value));
    *(impl->value_) = value;
    return *impl;
  }

  T& set_env_name(std::string_view env_name) {
    T* impl = static_cast<T*>(this);
    impl->env_name_ = std::string(env_name);
    return *impl;
  }

  T& set_greater_than_or_equal_to(bool greater_than_or_equal_to) {
    T* impl = static_cast<T*>(this);
    impl->greater_than_or_equal_to_ = greater_than_or_equal_to;
    return *impl;
  }

  T& set_less_than_or_equal_to(bool less_than_or_equal_to) {
    T* impl = static_cast<T*>(this);
    impl->less_than_or_equal_to_ = less_than_or_equal_to;
    return *impl;
  }
};

template <typename T>
class RangeFlag : public FlagBase, public RangeFlagBuilder<RangeFlag<T>, T> {
 public:
  using value_type = T;

  RangeFlag(T* value, const T& start, const T& end)
      : value_(value), start_(start), end_(end) {
    DCHECK_GE(end, start);
  }
  RangeFlag(const RangeFlag& other) = delete;
  RangeFlag& operator=(const RangeFlag& other) = delete;

 private:
  friend class RangeFlagBuilder<RangeFlag<T>, T>;
  FRIEND_TEST(FlagTest, ParseValue);

  void set_value(const T& value) {
    is_set_ = true;
    *value_ = value;
  }

  const T* value() const { return value_; }

  bool Contains(const T& value) {
    if (greater_than_or_equal_to_) {
      if (value < start_) return false;
    } else {
      if (value <= start_) return false;
    }
    if (less_than_or_equal_to_) {
      return value <= end_;
    } else {
      return value < end_;
    }
  }

  bool NeedsValue() const override { return !std::is_same<T, bool>::value; }
  bool ParseValue(std::string_view arg, std::string* reason) override;
  bool ParseValueFromEnvironment(std::string* reason) override;

  T* value_ = nullptr;
  T start_;
  T end_;
  bool greater_than_or_equal_to_ = false;
  bool less_than_or_equal_to_ = false;
  std::string env_name_;
};

template <typename T>
bool RangeFlag<T>::ParseValue(std::string_view arg, std::string* reason) {
  T value;
  if (FlagValueTraits<T>::ParseValue(arg, &value, reason)) {
    if (Contains(value)) {
      *value_ = std::move(value);
      is_set_ = true;
      return true;
    } else {
      *reason = absl::Substitute("$0 is not in range", arg);
    }
  }
  return false;
}

template <typename T>
bool RangeFlag<T>::ParseValueFromEnvironment(std::string* reason) {
  if (!env_name_.empty()) {
    std::string_view value;
    if (Environment::Get(env_name_, &value)) {
      return ParseValue(value, reason);
    }
  }
  return true;
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_FLAG_FLAG_H_
