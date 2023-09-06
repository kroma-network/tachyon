#ifndef TACHYON_BUILD_GENERATOR_UTIL_H_
#define TACHYON_BUILD_GENERATOR_UTIL_H_

#include <string>

#include "tachyon/base/functional/callback.h"

namespace tachyon::build {

std::string GenerateInsertionOperatorDeclaration(std::string_view type,
                                                 std::string_view name);

std::string GenerateInsertionOperatorDefinition(std::string_view type,
                                                std::string_view name,
                                                std::string_view impl);

std::string GenerateEqualityOpDeclarations(std::string_view type);

std::string GenerateEqualityOpDefinitions(
    std::string_view type,
    base::RepeatingCallback<std::string(std::string_view)> callback);

std::string GenerateComparisonOpDeclarations(std::string_view type);

std::string GenerateComparisonOpDefinitions(
    std::string_view type,
    base::RepeatingCallback<std::string(std::string_view)> callback);

}  // namespace tachyon::build

#endif  // TACHYON_BUILD_GENERATOR_UTIL_H_
