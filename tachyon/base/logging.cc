#include "tachyon/base/logging.h"

namespace tachyon {
namespace base {

// This is never instantiated, it's just used for EAT_STREAM_PARAMETERS to have
// an object of the correct type on the LHS of the unused part of the ternary
// operator.
std::ostream* g_swallow_stream;

}  // namespace base
}  // namespace tachyon