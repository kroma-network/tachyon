#include "tachyon/device/gpu/gpu_logging.h"

namespace google {

GpuLogMessage::GpuLogMessage(const char* file_path, int line,
                             LogSeverity severity, cudaError_t cuda_err)
    : LogMessage(file_path, line, severity), cuda_err_(cuda_err) {}

GpuLogMessage::~GpuLogMessage() {
  stream() << ": " << cudaGetErrorString(cuda_err_);
}

}  // namespace google
