#include "tachyon/build/generator_util.h"

#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"

namespace tachyon::build {

namespace {

std::string_view kEqualityOperators[] = {"==", "!="};
std::string_view kComparisonOperators[] = {"==", "!=", "<", "<=", ">", ">="};

std::string ToDecl(std::string_view sig) {
  return absl::Substitute("$0;", sig);
}

std::string ToDef(std::string_view sig, std::string_view impl) {
  std::vector<std::string> components = {
      absl::Substitute("$0 {", sig),
      std::string(impl),
      "}",
  };
  return absl::StrJoin(components, "\n");
}

std::string GenerateInsertionOpSignature(std::string_view type,
                                         std::string_view name) {
  return absl::Substitute(
      "std::ostream& operator<<(std::ostream& os, const $0& $1)", type, name);
}

std::string GenerateBinaryOpSignature(std::string_view op,
                                      std::string_view type) {
  return absl::Substitute("bool operator$0(const $1& a, const $1& b)", op,
                          type);
}

std::string GenerateOp(
    std::string_view type, absl::Span<std::string_view> operators,
    base::RepeatingCallback<std::string(std::string_view)> callback) {
  std::vector<std::string> components;
  for (size_t i = 0; i < operators.size(); ++i) {
    components.push_back(callback.Run(operators[i]));
    if (i != operators.size() - 1) {
      components.push_back("");
    }
  }
  return absl::StrJoin(components, "\n");
}

}  // namespace

std::string GenerateInsertionOperatorDeclaration(std::string_view type,
                                                 std::string_view name) {
  return ToDecl(GenerateInsertionOpSignature(type, name));
}

std::string GenerateInsertionOperatorDefinition(std::string_view type,
                                                std::string_view name,
                                                std::string_view impl) {
  return ToDef(GenerateInsertionOpSignature(type, name), impl);
}

std::string GenerateEqualityOpDeclarations(std::string_view type) {
  return GenerateOp(type, absl::MakeSpan(kEqualityOperators),
                    [type](std::string_view op) {
                      return ToDecl(GenerateBinaryOpSignature(op, type));
                    });
}

std::string GenerateEqualityOpDefinitions(
    std::string_view type,
    base::RepeatingCallback<std::string(std::string_view)> callback) {
  return GenerateOp(type, absl::MakeSpan(kEqualityOperators),
                    [type, callback](std::string_view op) {
                      return ToDef(GenerateBinaryOpSignature(op, type),
                                   callback.Run(op));
                    });
}

std::string GenerateComparisonOpDeclarations(std::string_view type) {
  return GenerateOp(type, absl::MakeSpan(kComparisonOperators),
                    [type](std::string_view op) {
                      return ToDecl(GenerateBinaryOpSignature(op, type));
                    });
}

std::string GenerateComparisonOpDefinitions(
    std::string_view type,
    base::RepeatingCallback<std::string(std::string_view)> callback) {
  return GenerateOp(type, absl::MakeSpan(kComparisonOperators),
                    [type, callback](std::string_view op) {
                      return ToDef(GenerateBinaryOpSignature(op, type),
                                   callback.Run(op));
                    });
}

}  // namespace tachyon::build
