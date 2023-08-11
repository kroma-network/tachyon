#ifndef TACHYON_BASE_CONSOLE_IOSTREAM_H_
#define TACHYON_BASE_CONSOLE_IOSTREAM_H_

#include <ostream>

#include "tachyon/base/console/console_stream.h"
#include "tachyon/export.h"

namespace tachyon::base::internal {

class TACHYON_EXPORT ConsoleErrStream {
 public:
  ConsoleErrStream();
  ConsoleErrStream(const ConsoleErrStream& other) = delete;
  ConsoleErrStream& operator=(const ConsoleErrStream& other) = delete;
  ~ConsoleErrStream();

  std::ostream& ostream() { return console_stream_.ostream(); }

 private:
  ConsoleStream console_stream_;
};

}  // namespace tachyon::base::internal

#define tachyon_cerr ::tachyon::base::internal::ConsoleErrStream().ostream()

#endif  // TACHYON_BASE_CONSOLE_IOSTREAM_H_
