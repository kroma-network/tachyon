// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// WARNING: You should *NOT* be using this class directly.  PlatformThread is
// the low-level platform-specific abstraction to the OS's threading interface.
// You should instead be using a message-loop driven Thread, see thread.h.

#ifndef TACHYON_BASE_THREADING_PLATFORM_THREAD_H_
#define TACHYON_BASE_THREADING_PLATFORM_THREAD_H_

#include <stddef.h>

#include <iosfwd>
#include <type_traits>
#include <optional>

#include "tachyon/export.h"
#include "tachyon/base/threading/platform_thread_ref.h"
#include "tachyon/base/time/time.h"
#include "tachyon/build/build_config.h"
#include "tachyon/base/message_loop/message_pump_type.h"

#if BUILDFLAG(IS_WIN)
#include "tachyon/base/win/windows_types.h"
#elif BUILDFLAG(IS_FUCHSIA)
#include <zircon/types.h>
#elif BUILDFLAG(IS_APPLE)
#include <mach/mach_types.h>
#elif BUILDFLAG(IS_POSIX)
#include <pthread.h>
#include <unistd.h>
#endif

namespace tachyon::base {

// Used for logging. Always an integer value.
#if BUILDFLAG(IS_WIN)
typedef DWORD PlatformThreadId;
#elif BUILDFLAG(IS_FUCHSIA)
typedef zx_handle_t PlatformThreadId;
#elif BUILDFLAG(IS_APPLE)
typedef mach_port_t PlatformThreadId;
#elif BUILDFLAG(IS_POSIX)
typedef pid_t PlatformThreadId;
#endif
static_assert(std::is_integral_v<PlatformThreadId>, "Always an integer value.");

// Used to operate on threads.
class PlatformThreadHandle {
 public:
#if BUILDFLAG(IS_WIN)
  typedef void* Handle;
#elif BUILDFLAG(IS_POSIX) || BUILDFLAG(IS_FUCHSIA)
  typedef pthread_t Handle;
#endif

  constexpr PlatformThreadHandle() : handle_(0) {}

  explicit constexpr PlatformThreadHandle(Handle handle) : handle_(handle) {}

  bool is_equal(const PlatformThreadHandle& other) const {
    return handle_ == other.handle_;
  }

  bool is_null() const {
    return !handle_;
  }

  Handle platform_handle() const {
    return handle_;
  }

 private:
  Handle handle_;
};

const PlatformThreadId kInvalidThreadId(0);

// Valid values for `thread_type` of Thread::Options, SimpleThread::Options,
// and SetCurrentThreadType(), listed in increasing order of importance.
//
// It is up to each platform-specific implementation what these translate to.
// Callers should avoid setting different ThreadTypes on different platforms
// (ifdefs) at all cost, instead the platform differences should be encoded in
// the platform-specific implementations. Some implementations may treat
// adjacent ThreadTypes in this enum as equivalent.
//
// Reach out to //base/task/OWNERS (scheduler-dev@chromium.org) before changing
// thread type assignments in your component, as such decisions affect the whole
// of Chrome.
//
// Refer to PlatformThreadTest.SetCurrentThreadTypeTest in
// platform_thread_unittest.cc for the most up-to-date state of each platform's
// handling of ThreadType.
enum class ThreadType : int {
  // Suitable for threads that have the least urgency and lowest priority, and
  // can be interrupted or delayed by other types.
  kBackground,
  // Suitable for threads that are less important than normal type, and can be
  // interrupted or delayed by threads with kDefault type.
  kUtility,
  // Suitable for threads that produce user-visible artifacts but aren't
  // latency sensitive. The underlying platform will try to be economic
  // in its usage of resources for this thread, if possible.
  kResourceEfficient,
  // Default type. The thread priority or quality of service will be set to
  // platform default. In Chrome, this is suitable for handling user
  // interactions (input), only display and audio can get a higher priority.
  kDefault,
  // Suitable for threads which are critical to compositing the foreground
  // content.
  kCompositing,
  // Suitable for display critical threads.
  kDisplayCritical,
  // Suitable for low-latency, glitch-resistant audio.
  kRealtimeAudio,
  kMaxValue = kRealtimeAudio,
};

TACHYON_EXPORT std::string ThreadTypeToString(ThreadType thread_type);

TACHYON_EXPORT std::ostream& operator<<(std::ostream& os,
                                        ThreadType thread_type);

// Cross-platform mapping of physical thread priorities. Used by tests to verify
// the underlying effects of SetCurrentThreadType.
enum class ThreadPriorityForTest : int {
  kBackground,
  kUtility,
  kNormal,
  // The priority obtained via ThreadType::kDisplayCritical (and potentially
  // other ThreadTypes).
  kDisplay,
  kRealtimeAudio,
  kMaxValue = kRealtimeAudio,
};

#if BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS)
class ThreadTypeDelegate;
#endif

// A namespace for low-level thread functions.
class TACHYON_EXPORT PlatformThread {
 public:
  // Implement this interface to run code on a background thread.  Your
  // ThreadMain method will be called on the newly created thread.
  class TACHYON_EXPORT Delegate {
   public:
#if BUILDFLAG(IS_APPLE)
    // The interval at which the thread expects to have work to do. Zero if
    // unknown. (Example: audio buffer duration for real-time audio.) Is used to
    // optimize the thread real-time behavior. Is called on the newly created
    // thread before ThreadMain().
    virtual TimeDelta GetRealtimePeriod();
#endif

    virtual void ThreadMain() = 0;

   protected:
    virtual ~Delegate() = default;
  };

  PlatformThread() = delete;
  PlatformThread(const PlatformThread&) = delete;
  PlatformThread& operator=(const PlatformThread&) = delete;

  // Gets the current thread id, which may be useful for logging purposes.
  static PlatformThreadId CurrentId();

  // Gets the current thread reference, which can be used to check if
  // we're on the right thread quickly.
  static PlatformThreadRef CurrentRef();

  // Get the handle representing the current thread. On Windows, this is a
  // pseudo handle constant which will always represent the thread using it and
  // hence should not be shared with other threads nor be used to differentiate
  // the current thread from another.
  static PlatformThreadHandle CurrentHandle();

  // Yield the current thread so another thread can be scheduled.
  //
  // Note: this is likely not the right call to make in most situations. If this
  // is part of a spin loop, consider base::Lock, which likely has better tail
  // latency. Yielding the thread has different effects depending on the
  // platform, system load, etc., and can result in yielding the CPU for less
  // than 1us, or many tens of ms.
  static void YieldCurrentThread();

  // Sleeps for the specified duration (real-time; ignores time overrides).
  // Note: The sleep duration may be in base::Time or base::TimeTicks, depending
  // on platform. If you're looking to use this in unit tests testing delayed
  // tasks, this will be unreliable - instead, use
  // base::test::TaskEnvironment with MOCK_TIME mode.
  static void Sleep(base::TimeDelta duration);

  // Sets the thread name visible to debuggers/tools. This will try to
  // initialize the context for current thread unless it's a WorkerThread.
  static void SetName(const std::string& name);

  // Gets the thread name, if previously set by SetName.
  static const char* GetName();

  // Creates a new thread.  The `stack_size` parameter can be 0 to indicate
  // that the default stack size should be used.  Upon success,
  // `*thread_handle` will be assigned a handle to the newly created thread,
  // and `delegate`'s ThreadMain method will be executed on the newly created
  // thread.
  // NOTE: When you are done with the thread handle, you must call Join to
  // release system resources associated with the thread.  You must ensure that
  // the Delegate object outlives the thread.
  static bool Create(size_t stack_size,
                     Delegate* delegate,
                     PlatformThreadHandle* thread_handle) {
    return CreateWithType(stack_size, delegate, thread_handle,
                          ThreadType::kDefault);
  }

  // CreateWithType() does the same thing as Create() except the priority and
  // possibly the QoS of the thread is set based on `thread_type`.
  // `pump_type_hint` must be provided if the thread will be using a
  // MessagePumpForUI or MessagePumpForIO as this affects the application of
  // `thread_type`.
  static bool CreateWithType(
      size_t stack_size,
      Delegate* delegate,
      PlatformThreadHandle* thread_handle,
      ThreadType thread_type,
      MessagePumpType pump_type_hint = MessagePumpType::DEFAULT);

  // CreateNonJoinable() does the same thing as Create() except the thread
  // cannot be Join()'d.  Therefore, it also does not output a
  // PlatformThreadHandle.
  static bool CreateNonJoinable(size_t stack_size, Delegate* delegate);

  // CreateNonJoinableWithType() does the same thing as CreateNonJoinable()
  // except the type of the thread is set based on `type`. `pump_type_hint` must
  // be provided if the thread will be using a MessagePumpForUI or
  // MessagePumpForIO as this affects the application of `thread_type`.
  static bool CreateNonJoinableWithType(
      size_t stack_size,
      Delegate* delegate,
      ThreadType thread_type,
      MessagePumpType pump_type_hint = MessagePumpType::DEFAULT);

  // Joins with a thread created via the Create function.  This function blocks
  // the caller until the designated thread exits.  This will invalidate
  // `thread_handle`.
  static void Join(PlatformThreadHandle thread_handle);

  // Detaches and releases the thread handle. The thread is no longer joinable
  // and `thread_handle` is invalidated after this call.
  static void Detach(PlatformThreadHandle thread_handle);

  // Returns true if SetCurrentThreadType() should be able to change the type
  // of a thread in current process from `from` to `to`.
  static bool CanChangeThreadType(ThreadType from, ThreadType to);

  // Declares the type of work running on the current thread. This will affect
  // things like thread priority and thread QoS (Quality of Service) to the best
  // of the current platform's abilities.
  static void SetCurrentThreadType(ThreadType thread_type);

  // Get the last `thread_type` set by SetCurrentThreadType, no matter if the
  // underlying priority successfully changed or not.
  static ThreadType GetCurrentThreadType();

  // Returns a realtime period provided by `delegate`.
  static TimeDelta GetRealtimePeriod(Delegate* delegate);

  // Returns the override of task leeway if any.
  static std::optional<TimeDelta> GetThreadLeewayOverride();

#if BUILDFLAG(IS_LINUX) || BUILDFLAG(IS_CHROMEOS)
  // Sets a delegate which handles thread type changes for this process. This
  // must be externally synchronized with any call to SetCurrentThreadType.
  static void SetThreadTypeDelegate(ThreadTypeDelegate* delegate);

  // Toggles a specific thread's type at runtime. This can be used to
  // change the priority of a thread in a different process and will fail
  // if the calling process does not have proper permissions. The
  // SetCurrentThreadType() function above is preferred in favor of
  // security but on platforms where sandboxed processes are not allowed to
  // change priority this function exists to allow a non-sandboxed process
  // to change the priority of sandboxed threads for improved performance.
  // Warning: Don't use this for a main thread because that will change the
  // whole thread group's (i.e. process) priority.
  static void SetThreadType(PlatformThreadId process_id,
                            PlatformThreadId thread_id,
                            ThreadType thread_type);
#endif

#if BUILDFLAG(IS_CHROMEOS) || BUILDFLAG(IS_APPLE)
  // Signals that the feature list has been initialized which allows to check
  // the feature's value now and initialize state. This prevents race
  // conditions where the feature is being checked while it is being
  // initialized, which can cause a crash.
  static void InitFeaturesPostFieldTrial();
#endif

  // Returns the default thread stack size set by chrome. If we do not
  // explicitly set default size then returns 0.
  static size_t GetDefaultThreadStackSize();

#if BUILDFLAG(IS_APPLE)
  // Stores the period value in TLS.
  static void SetCurrentThreadRealtimePeriodValue(TimeDelta realtime_period);
#endif

  static ThreadPriorityForTest GetCurrentThreadPriorityForTest();
};

namespace internal {

void SetCurrentThreadType(ThreadType thread_type,
                          MessagePumpType pump_type_hint);

void SetCurrentThreadTypeImpl(ThreadType thread_type,
                              MessagePumpType pump_type_hint);

}  // namespace internal

}  // namespace tachyon::base

#endif  // TACHYON_BASE_THREADING_PLATFORM_THREAD_H_
