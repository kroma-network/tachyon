#ifndef TACHYON_DEVICE_GPU_GPU_LOGGING_H_
#define TACHYON_DEVICE_GPU_GPU_LOGGING_H_

#include "tachyon/base/logging.h"
#include "tachyon/device/gpu/gpu_device_functions.h"

namespace google {

class TACHYON_EXPORT GpuLogMessage : public LogMessage {
 public:
  GpuLogMessage(const char* file_path, int line, LogSeverity severity,
                gpuError_t gpu_err);

  GpuLogMessage(const GpuLogMessage&) = delete;
  GpuLogMessage& operator=(const GpuLogMessage&) = delete;

  ~GpuLogMessage();

 private:
  gpuError_t gpu_err_;
};

}  // namespace google

#if DCHECK_IS_ON()
#define GPU_DVLOG_IS_ON(verbose_level) VLOG_IS_ON(verbose_level)
#else
#define GPU_DVLOG_IS_ON(verbose_level) 0
#endif

#define GPU_LOG_STREAM(severity, gpu_err) \
  COMPACT_GOOGLE_LOG_EX_##severity(GpuLogMessage, gpu_err).stream()
#define GPU_VLOG_STREAM(verbose_level, gpu_err) \
  ::google::GpuLogMessage(__FILE__, __LINE__, -verbose_level, gpu_err).stream()

#define GPU_LOG(severity, gpu_err) \
  LAZY_STREAM(GPU_LOG_STREAM(severity, gpu_err), LOG_IS_ON(severity))
#define GPU_LOG_IF(severity, condition, gpu_err) \
  LAZY_STREAM(GPU_LOG_STREAM(severity, gpu_err), \
              LOG_IS_ON(severity) && (condition))

#define GPU_VLOG(verbose_level, gpu_err)               \
  LAZY_STREAM(GPU_VLOG_STREAM(verbose_level, gpu_err), \
              VLOG_IS_ON(verbose_level))
#define GPU_VLOG_IF(verbose_level, condition, gpu_err) \
  LAZY_STREAM(GPU_VLOG_STREAM(verbose_level, gpu_err), \
              VLOG_IS_ON(verbose_level) && (condition))

#define GPU_CHECK(condition, gpu_err)                       \
  LAZY_STREAM(GPU_LOG_STREAM(FATAL, gpu_err), !(condition)) \
      << "Check failed: " #condition << ". "

#define GPU_DLOG(severity, gpu_err) \
  LAZY_STREAM(GPU_LOG_STREAM(severity, gpu_err), DLOG_IS_ON(severity))
#define GPU_DLOG_IF(severity, condition, gpu_err) \
  LAZY_STREAM(GPU_LOG_STREAM(severity, gpu_err),  \
              DLOG_IS_ON(severity) && (condition))

#define GPU_DVLOG(verbose_level, gpu_err)              \
  LAZY_STREAM(GPU_VLOG_STREAM(verbose_level, gpu_err), \
              GPU_DVLOG_IS_ON(verbose_level))
#define GPU_DVLOG_IF(verbose_level, condition, gpu_err) \
  LAZY_STREAM(GPU_VLOG_STREAM(verbose_level, gpu_err),  \
              GPU_DVLOG_IS_ON(verbose_level) && (condition))

#define GPU_DCHECK(condition, gpu_err)                                        \
  LAZY_STREAM(GPU_LOG_STREAM(FATAL, gpu_err), DCHECK_IS_ON() && !(condition)) \
      << "Check failed: " #condition << ". "

#endif  // TACHYON_DEVICE_GPU_GPU_LOGGING_H_
