// Copyright (c) 2020 The Console Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef TACHYON_BASE_FLAG_FLAG_PARSER_H_
#define TACHYON_BASE_FLAG_FLAG_PARSER_H_

#include <memory>
#include <type_traits>
#include <vector>

#include "tachyon/base/files/file_path.h"
#include "tachyon/base/flag/flag.h"
#include "tachyon/export.h"

namespace tachyon::base {

class TACHYON_EXPORT FlagParserBase {
 public:
  FlagParserBase();
  FlagParserBase(const FlagParserBase& other) = delete;
  FlagParserBase& operator=(const FlagParserBase& other) = delete;
  virtual ~FlagParserBase();

  const std::vector<std::unique_ptr<FlagBase>>* flags() const {
    return &flags_;
  }

  template <typename T, typename value_type = typename Flag<T>::value_type>
  T& AddFlag(value_type* value) {
    std::unique_ptr<FlagBase> flag(new T(value));
    flags_.push_back(std::move(flag));
    return *reinterpret_cast<T*>(flags_.back().get());
  }

  template <typename T,
            typename ParseValueCallback = typename Flag<T>::ParseValueCallback>
  T& AddFlag(ParseValueCallback parse_value_callback) {
    std::unique_ptr<FlagBase> flag(new T(parse_value_callback));
    flags_.push_back(std::move(flag));
    return *reinterpret_cast<T*>(flags_.back().get());
  }

  template <typename T,
            typename value_type = typename ChoicesFlag<T>::value_type>
  T& AddFlag(value_type* value, const std::vector<value_type>& choices) {
    std::unique_ptr<FlagBase> flag(new T(value, choices));
    flags_.push_back(std::move(flag));
    return *reinterpret_cast<T*>(flags_.back().get());
  }

  template <typename T,
            typename value_type = typename ChoicesFlag<T>::value_type>
  T& AddFlag(value_type* value, std::vector<value_type>&& choices) {
    std::unique_ptr<FlagBase> flag(new T(value, std::move(choices)));
    flags_.push_back(std::move(flag));
    return *reinterpret_cast<T*>(flags_.back().get());
  }

  template <typename T, typename value_type = typename RangeFlag<T>::value_type>
  T& AddFlag(value_type* value, const value_type& start,
             const value_type& end) {
    std::unique_ptr<FlagBase> flag(new T(value, start, end));
    flags_.push_back(std::move(flag));
    return *reinterpret_cast<T*>(flags_.back().get());
  }

  FlagBase& AddFlag(std::unique_ptr<FlagBase> flag_base);

  SubParser& AddSubParser();

 protected:
  FRIEND_TEST(FlagParserTest, ValidateInternally);

  struct Context {
    FlagParser* parser;
    int current_idx;
    int argc;
    char** argv;
    int* unknown_argc = nullptr;
    std::vector<char*> unknown_argv;
    std::vector<std::string>* forward_argv = nullptr;

    Context(FlagParser* parser, int current_idx, int argc, char** argv);
    ~Context();

    std::string_view current() const;
    bool ConsumeEqualOrProceed(std::string_view* arg);
    void Proceed();
    bool HasArg() const;
    void FillUnknownArgs() const;
  };

  bool Parse(Context& ctx, std::string* error);

  bool ValidateInternally(std::string* error) const;

  // Internally it measures Levenshtein distance among arguments.
  bool FindTheMostSimilarFlag(std::string_view input, std::string_view* output);

 protected:
  std::vector<std::unique_ptr<FlagBase>> flags_;
};

class TACHYON_EXPORT FlagParser : public FlagParserBase {
 public:
  FlagParser();
  FlagParser(const FlagParser& other) = delete;
  FlagParser& operator=(const FlagParser& other) = delete;
  ~FlagParser();

  void set_program_path(const FilePath& program_path) {
    program_path_ = program_path;
  }
  const FilePath& program_path() const { return program_path_; }

  // Sames as ParseWithForward(argc, argv, nullptr, error).
  bool Parse(int argc, char** argv, std::string* error);

  // Sames as ParseKnownWithForward(argc, argv, nullptr, error).
  // It only parses known arguments. After parsing, |argc| and |argv| are
  // updated to match the unknown arguments.
  bool ParseKnown(int* argc, char** argv, std::string* error);

  bool ParseWithForward(int argc, char** argv,
                        std::vector<std::string>* forward, std::string* error);

  bool ParseKnownWithForward(int* argc, char** argv,
                             std::vector<std::string>* forward,
                             std::string* error);

  // It is marked virtual so that users can make custom help messages.
  virtual std::string help_message();

  virtual bool Validate(std::string* error) { return true; }

 private:
  bool PreParse(int argc, char** argv, std::string* error);

  FilePath program_path_;
};

class TACHYON_EXPORT SubParser : public FlagBase,
                                 public FlagParserBase,
                                 public FlagBaseBuilder<SubParser> {
 public:
  SubParser();
  ~SubParser();
  SubParser(const FlagBase& other) = delete;
  SubParser& operator=(const FlagBase& other) = delete;

  // FlagBase methods
  bool IsSubParser() const override;
  bool NeedsValue() const override;
  bool ParseValue(std::string_view arg, std::string* reason) override;

  SubParser& set_is_set(bool* is_set) {
    is_set_ = is_set;
    return *this;
  }

 private:
  friend class FlagParserBase;

  bool* is_set_ = nullptr;
};

}  // namespace tachyon::base

#endif  // TACHYON_BASE_FLAG_FLAG_PARSER_H_
