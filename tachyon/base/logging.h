#ifndef TACHYON_BASE_LOGGING_H_
#define TACHYON_BASE_LOGGING_H_

#include <ostream>

#include "tachyon/export.h"

// These are codes to minimize gaps between glog and chromium logging.
// Followings are modified and taken from chromium/base/logging.
// - LAZY_STREAM
// - EAT_STREAM_PARAMETERS
// - ANALYZER_*

#define GLOG_NO_ABBREVIATED_SEVERITIES
#include "glog/logging.h"
#include "glog/raw_logging.h"

namespace tachyon::base {

TACHYON_EXPORT extern std::ostream* g_swallow_stream;

}  // namespace tachyon::base

// Helper macro which avoids evaluating the arguments to a stream if
// the condition doesn't hold. Condition is evaluated once and only once.
#define LAZY_STREAM(stream, condition) \
  !(condition) ? (void)0 : ::google::LogMessageVoidify() & (stream)

// Note that g_swallow_stream is used instead of an arbitrary LOG() stream to
// avoid the creation of an object with a non-trivial destructor (LogMessage).
// On MSVC x86 (checked on 2015 Update 3), this causes a few additional
// pointless instructions to be emitted even at full optimization level, even
// though the : arm of the ternary operator is clearly never executed. Using a
// simpler object to be &'d with Voidify() avoids these extra instructions.
// Using a simpler POD object with a templated operator<< also works to avoid
// these instructions. However, this causes warnings on statically defined
// implementations of operator<<(std::ostream, ...) in some .cc files, because
// they become defined-but-unreferenced functions. A reinterpret_cast of 0 to an
// ostream* also is not suitable, because some compilers warn of undefined
// behavior.
#define EAT_STREAM_PARAMETERS \
  true ? (void)0              \
       : ::google::LogMessageVoidify() & (*::tachyon::base::g_swallow_stream)

#define NOTREACHED() CHECK(false)

#define VPLOG(verboselevel) PLOG_IF(INFO, VLOG_IS_ON(verboselevel))

#define VPLOG_IF(verboselevel, condition) \
  PLOG_IF(INFO, VLOG_IS_ON(verboselevel) && (condition))

#define DPLOG(severity) LAZY_STREAM(PLOG(severity), DCHECK_IS_ON())
#define DVPLOG(verboselevel) \
  LAZY_STREAM(PLOG(INFO), DCHECK_IS_ON() && VLOG_IS_ON(verboselevel))

#if DCHECK_IS_ON()

#define DPLOG_IF(severity, condition) PLOG_IF(severity, condition)
#define DVLOG_IF(verboselevel, condition) VLOG_IF(verboselevel, condition)
#define DVPLOG_IF(verboselevel, condition) VPLOG_IF(verboselevel, condition)

#define DPCHECK(condition) PCHECK(condition)

#else  // !DCHECK_IS_ON()

#define DPLOG_IF(severity, condition) EAT_STREAM_PARAMETERS
#define DVLOG_IF(verboselevel, condition) EAT_STREAM_PARAMETERS
#define DVPLOG_IF(verboselevel, condition) EAT_STREAM_PARAMETERS

#define DPCHECK(condition) EAT_STREAM_PARAMETERS

#endif

#if defined(COMPILER_GCC)
// On Linux, with GCC, we can use __PRETTY_FUNCTION__ to get the demangled name
// of the current function in the NOTIMPLEMENTED message.
#define NOTIMPLEMENTED_MSG "Not implemented reached in " << __PRETTY_FUNCTION__
#else
#define NOTIMPLEMENTED_MSG "NOT IMPLEMENTED"
#endif

#define NOTIMPLEMENTED() DLOG(ERROR) << NOTIMPLEMENTED_MSG
#define NOTIMPLEMENTED_LOG_ONCE()                      \
  do {                                                 \
    static bool logged_once = false;                   \
    LOG_IF(ERROR, !logged_once) << NOTIMPLEMENTED_MSG; \
    logged_once = true;                                \
  } while (0);                                         \
  EAT_STREAM_PARAMETERS

#endif  // TACHYON_BASE_LOGGING_H_
