#include "tachyon/device/gpu/gpu_logging.h"

namespace google {

GpuLogMessage::GpuLogMessage(const char* file_path, int line,
                             LogSeverity severity, gpuError_t gpu_err)
    : LogMessage(file_path, line, severity), gpu_err_(gpu_err) {}

GpuLogMessage::~GpuLogMessage() {
  stream() << ": " << gpuGetErrorString(gpu_err_);
}

}  // namespace google
