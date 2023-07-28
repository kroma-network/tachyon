// Copyright 2012 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/threading/platform_thread.h"

#import <Foundation/Foundation.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/thread_policy.h>
#include <mach/thread_switch.h>
#include <stddef.h>
#include <sys/resource.h>

#include <algorithm>
#include <atomic>

#include "tachyon/base/logging.h"
// #include "tachyon/base/mac/foundation_util.h"
// #include "tachyon/base/mac/mac_util.h"
// #include "tachyon/base/mac/mach_logging.h"
// #include "tachyon/base/metrics/histogram_functions.h"
// #include "tachyon/base/threading/thread_id_name_manager.h"
// #include "tachyon/base/threading/threading_features.h"
#include "tachyon/build/build_config.h"

#if !defined(__has_feature) || !__has_feature(objc_arc)
#error "This file requires ARC support."
#endif

namespace tachyon::base {

namespace {
NSString* const kThreadPriorityForTestKey = @"CrThreadPriorityForTestKey";
NSString* const kRealtimePeriodNsKey = @"CrRealtimePeriodNsKey";
}  // namespace

// If Cocoa is to be used on more than one thread, it must know that the
// application is multithreaded.  Since it's possible to enter Cocoa code
// from threads created by pthread_thread_create, Cocoa won't necessarily
// be aware that the application is multithreaded.  Spawning an NSThread is
// enough to get Cocoa to set up for multithreaded operation, so this is done
// if necessary before pthread_thread_create spawns any threads.
//
// http://developer.apple.com/documentation/Cocoa/Conceptual/Multithreading/CreatingThreads/chapter_4_section_4.html
void InitThreading() {
  static BOOL multithreaded = [NSThread isMultiThreaded];
  if (!multithreaded) {
    // +[NSObject class] is idempotent.
    @autoreleasepool {
      [NSThread detachNewThreadSelector:@selector(class)
                               toTarget:[NSObject class]
                             withObject:nil];
      multithreaded = YES;

      DCHECK([NSThread isMultiThreaded]);
    }
  }
}

TimeDelta PlatformThreadBase::Delegate::GetRealtimePeriod() {
  return TimeDelta();
}

// static
void PlatformThreadBase::YieldCurrentThread() {
  // Don't use sched_yield(), as it can lead to 10ms delays.
  //
  // This only depresses the thread priority for 1ms, which is more in line
  // with what calling code likely wants. See this bug in webkit for context:
  // https://bugs.webkit.org/show_bug.cgi?id=204871
  mach_msg_timeout_t timeout_ms = 1;
  thread_switch(MACH_PORT_NULL, SWITCH_OPTION_DEPRESS, timeout_ms);
}

// static
void PlatformThreadBase::SetName(const std::string& name) {
  SetNameCommon(name);

  // macOS does not expose the length limit of the name, so hardcode it.
  const int kMaxNameLength = 63;
  std::string shortened_name = name.substr(0, kMaxNameLength);
  // pthread_setname() fails (harmlessly) in the sandbox, ignore when it does.
  // See http://crbug.com/47058
  pthread_setname_np(shortened_name.c_str());
}

/*
TODO(chokobole):
// Whether optimized realt-time thread config should be used for audio.
BASE_FEATURE(kOptimizedRealtimeThreadingMac,
             "OptimizedRealtimeThreadingMac",
#if BUILDFLAG(IS_MAC)
             FEATURE_ENABLED_BY_DEFAULT
#else
             FEATURE_DISABLED_BY_DEFAULT
#endif
);

const Feature kUserInteractiveCompositingMac{"UserInteractiveCompositingMac",
                                             FEATURE_DISABLED_BY_DEFAULT};

namespace {

bool IsOptimizedRealtimeThreadingMacEnabled() {
  return FeatureList::IsEnabled(kOptimizedRealtimeThreadingMac);
}

}  // namespace

// Fine-tuning optimized real-time thread config:
// Whether or not the thread should be preemptible.
const FeatureParam<bool> kOptimizedRealtimeThreadingMacPreemptible{
    &kOptimizedRealtimeThreadingMac, "preemptible", true};
// Portion of the time quantum the thread is expected to be busy, (0, 1].
const FeatureParam<double> kOptimizedRealtimeThreadingMacBusy{
    &kOptimizedRealtimeThreadingMac, "busy", 0.5};
// Maximum portion of the time quantum the thread is expected to be busy,
// (kOptimizedRealtimeThreadingMacBusy, 1].
const FeatureParam<double> kOptimizedRealtimeThreadingMacBusyLimit{
    &kOptimizedRealtimeThreadingMac, "busy_limit", 1.0};
std::atomic<bool> g_user_interactive_compositing(
    kUserInteractiveCompositingMac.default_state == FEATURE_ENABLED_BY_DEFAULT);

namespace {

struct TimeConstraints {
  bool preemptible{kOptimizedRealtimeThreadingMacPreemptible.default_value};
  double busy{kOptimizedRealtimeThreadingMacBusy.default_value};
  double busy_limit{kOptimizedRealtimeThreadingMacBusyLimit.default_value};

  static TimeConstraints ReadFromFeatureParams() {
    double busy_limit = kOptimizedRealtimeThreadingMacBusyLimit.Get();
    return TimeConstraints{
        kOptimizedRealtimeThreadingMacPreemptible.Get(),
        std::min(busy_limit, kOptimizedRealtimeThreadingMacBusy.Get()),
        busy_limit};
  }
};

// Use atomics to access FeatureList values when setting up a thread, since
// there are cases when FeatureList initialization is not synchronized with
// PlatformThread creation.
std::atomic<bool> g_use_optimized_realtime_threading(
    kOptimizedRealtimeThreadingMac.default_state == FEATURE_ENABLED_BY_DEFAULT);
std::atomic<TimeConstraints> g_time_constraints;

}  // namespace
*/

// static
void PlatformThreadApple::InitFeaturesPostFieldTrial() {
  /*
  TODO(chokobole):
  // A DCHECK is triggered on FeatureList initialization if the state of a
  // feature has been checked before. To avoid triggering this DCHECK in unit
  // tests that call this before initializing the FeatureList, only check the
  // state of the feature if the FeatureList is initialized.
  if (FeatureList::GetInstance()) {
    g_time_constraints.store(TimeConstraints::ReadFromFeatureParams());
    g_use_optimized_realtime_threading.store(
        IsOptimizedRealtimeThreadingMacEnabled());
    g_user_interactive_compositing.store(
        FeatureList::IsEnabled(kUserInteractiveCompositingMac));
  }
  */
}

// static
void PlatformThreadApple::SetCurrentThreadRealtimePeriodValue(
    TimeDelta realtime_period) {
  /*
  TODO(chokobole):
  if (g_use_optimized_realtime_threading.load()) {
    NSThread.currentThread.threadDictionary[kRealtimePeriodNsKey] =
        @(realtime_period.InNanoseconds());
  }
  */
}

namespace {
/*
TODO(chokobole):
TimeDelta GetCurrentThreadRealtimePeriod() {
  NSNumber* period = mac::ObjCCast<NSNumber>(
      NSThread.currentThread.threadDictionary[kRealtimePeriodNsKey]);

  return period ? Nanoseconds(period.longLongValue) : TimeDelta();
}

// Calculates time constraints for THREAD_TIME_CONSTRAINT_POLICY.
// |realtime_period| is used as a base if it's non-zero.
// Otherwise we fall back to empirical values.
thread_time_constraint_policy_data_t GetTimeConstraints(
    TimeDelta realtime_period) {
  thread_time_constraint_policy_data_t time_constraints;
  mach_timebase_info_data_t tb_info;
  mach_timebase_info(&tb_info);

  if (!realtime_period.is_zero()) {
    // Limit the lowest value to 2.9 ms we used to have historically. The lower
    // the period, the more CPU frequency may go up, and we don't want to risk
    // worsening the thermal situation.
    uint32_t abs_realtime_period = saturated_cast<uint32_t>(
        std::max(realtime_period.InNanoseconds(), 2900000LL) *
        (double(tb_info.denom) / tb_info.numer));
    TimeConstraints config = g_time_constraints.load();
    time_constraints.period = abs_realtime_period;
    time_constraints.constraint = std::min(
        abs_realtime_period, uint32_t(abs_realtime_period * config.busy_limit));
    time_constraints.computation =
        std::min(time_constraints.constraint,
                 uint32_t(abs_realtime_period * config.busy));
    time_constraints.preemptible = config.preemptible ? YES : NO;
    return time_constraints;
  }

  // Empirical configuration.

  // Define the guaranteed and max fraction of time for the audio thread.
  // These "duty cycle" values can range from 0 to 1.  A value of 0.5
  // means the scheduler would give half the time to the thread.
  // These values have empirically been found to yield good behavior.
  // Good means that audio performance is high and other threads won't starve.
  const double kGuaranteedAudioDutyCycle = 0.75;
  const double kMaxAudioDutyCycle = 0.85;

  // Define constants determining how much time the audio thread can
  // use in a given time quantum.  All times are in milliseconds.

  // About 128 frames @44.1KHz
  const double kTimeQuantum = 2.9;

  // Time guaranteed each quantum.
  const double kAudioTimeNeeded = kGuaranteedAudioDutyCycle * kTimeQuantum;

  // Maximum time each quantum.
  const double kMaxTimeAllowed = kMaxAudioDutyCycle * kTimeQuantum;

  // Get the conversion factor from milliseconds to absolute time
  // which is what the time-constraints call needs.
  double ms_to_abs_time = double(tb_info.denom) / tb_info.numer * 1000000;

  time_constraints.period = kTimeQuantum * ms_to_abs_time;
  time_constraints.computation = kAudioTimeNeeded * ms_to_abs_time;
  time_constraints.constraint = kMaxTimeAllowed * ms_to_abs_time;
  time_constraints.preemptible = 0;
  return time_constraints;
}

// Enables time-constraint policy and priority suitable for low-latency,
// glitch-resistant audio.
void SetPriorityRealtimeAudio(TimeDelta realtime_period) {
  // Increase thread priority to real-time.

  // Please note that the thread_policy_set() calls may fail in
  // rare cases if the kernel decides the system is under heavy load
  // and is unable to handle boosting the thread priority.
  // In these cases we just return early and go on with life.

  mach_port_t mach_thread_id =
      pthread_mach_thread_np(PlatformThread::CurrentHandle().platform_handle());

  // Make thread fixed priority.
  thread_extended_policy_data_t policy;
  policy.timeshare = 0;  // Set to 1 for a non-fixed thread.
  kern_return_t result = thread_policy_set(
      mach_thread_id, THREAD_EXTENDED_POLICY,
      reinterpret_cast<thread_policy_t>(&policy), THREAD_EXTENDED_POLICY_COUNT);
  if (result != KERN_SUCCESS) {
    MACH_DVLOG(1, result) << "thread_policy_set";
    return;
  }

  // Set to relatively high priority.
  thread_precedence_policy_data_t precedence;
  precedence.importance = 63;
  result = thread_policy_set(mach_thread_id, THREAD_PRECEDENCE_POLICY,
                             reinterpret_cast<thread_policy_t>(&precedence),
                             THREAD_PRECEDENCE_POLICY_COUNT);
  if (result != KERN_SUCCESS) {
    MACH_DVLOG(1, result) << "thread_policy_set";
    return;
  }

  // Most important, set real-time constraints.

  thread_time_constraint_policy_data_t time_constraints =
      GetTimeConstraints(realtime_period);

  result =
      thread_policy_set(mach_thread_id, THREAD_TIME_CONSTRAINT_POLICY,
                        reinterpret_cast<thread_policy_t>(&time_constraints),
                        THREAD_TIME_CONSTRAINT_POLICY_COUNT);
  MACH_DVLOG_IF(1, result != KERN_SUCCESS, result) << "thread_policy_set";
  return;
}
*/

}  // anonymous namespace

// static
bool PlatformThreadBase::CanChangeThreadType(ThreadType from, ThreadType to) {
  return true;
}

namespace internal {

void SetCurrentThreadTypeImpl(ThreadType thread_type,
                              MessagePumpType pump_type_hint) {
  // Changing the priority of the main thread causes performance
  // regressions. https://crbug.com/601270
  // TODO(1280764): Remove this check. kCompositing is the default on Mac, so
  // this check is counter intuitive.
  if ([[NSThread currentThread] isMainThread] &&
      thread_type >= ThreadType::kCompositing) {
    DCHECK(thread_type == ThreadType::kDefault ||
           thread_type == ThreadType::kCompositing);
    return;
  }

  ThreadPriorityForTest priority = ThreadPriorityForTest::kNormal;
  switch (thread_type) {
    case ThreadType::kBackground:
      priority = ThreadPriorityForTest::kBackground;
      pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
      break;
    case ThreadType::kUtility:
      priority = ThreadPriorityForTest::kUtility;
      pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
      break;
    case ThreadType::kResourceEfficient:
      priority = ThreadPriorityForTest::kUtility;
      pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
      break;
    case ThreadType::kDefault:
      priority = ThreadPriorityForTest::kNormal;
      pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
      break;
    case ThreadType::kCompositing:
      /*
      TODO(chokobole):
      if (g_user_interactive_compositing.load(std::memory_order_relaxed)) {
        priority = ThreadPriorityForTest::kDisplay;
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
      } else {
        priority = ThreadPriorityForTest::kNormal;
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
      }
      */
      break;
    case ThreadType::kDisplayCritical: {
      priority = ThreadPriorityForTest::kDisplay;
      pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
      break;
    }
    case ThreadType::kRealtimeAudio:
      priority = ThreadPriorityForTest::kRealtimeAudio;
      // TODO(chokobole):
      // SetPriorityRealtimeAudio(GetCurrentThreadRealtimePeriod());
      DCHECK_EQ([NSThread.currentThread threadPriority], 1.0);
      break;
  }

  NSThread.currentThread.threadDictionary[kThreadPriorityForTestKey] =
      @(static_cast<int>(priority));
}

}  // namespace internal

/*
TODO(chokobole):
// static
ThreadPriorityForTest PlatformThreadBase::GetCurrentThreadPriorityForTest() {
  NSNumber* priority = base::mac::ObjCCast<NSNumber>(
      NSThread.currentThread.threadDictionary[kThreadPriorityForTestKey]);

  if (!priority)
    return ThreadPriorityForTest::kNormal;

  ThreadPriorityForTest thread_priority =
      static_cast<ThreadPriorityForTest>(priority.intValue);
  DCHECK_GE(thread_priority, ThreadPriorityForTest::kBackground);
  DCHECK_LE(thread_priority, ThreadPriorityForTest::kMaxValue);
  return thread_priority;
}
*/

size_t GetDefaultThreadStackSize(const pthread_attr_t& attributes) {
#if BUILDFLAG(IS_IOS)
#if BUILDFLAG(USE_BLINK)
  // For iOS 512kB (the default) isn't sufficient, but using the code
  // for macOS below will return 8MB. So just be a little more conservative
  // and return 1MB for now.
  return 1024 * 1024;
#else
  return 0;
#endif
#else
  // The macOS default for a pthread stack size is 512kB.
  // Libc-594.1.4/pthreads/pthread.c's pthread_attr_init uses
  // DEFAULT_STACK_SIZE for this purpose.
  //
  // 512kB isn't quite generous enough for some deeply recursive threads that
  // otherwise request the default stack size by specifying 0. Here, adopt
  // glibc's behavior as on Linux, which is to use the current stack size
  // limit (ulimit -s) as the default stack size. See
  // glibc-2.11.1/nptl/nptl-init.c's __pthread_initialize_minimal_internal. To
  // avoid setting the limit below the macOS default or the minimum usable
  // stack size, these values are also considered. If any of these values
  // can't be determined, or if stack size is unlimited (ulimit -s unlimited),
  // stack_size is left at 0 to get the system default.
  //
  // macOS normally only applies ulimit -s to the main thread stack. On
  // contemporary macOS and Linux systems alike, this value is generally 8MB
  // or in that neighborhood.
  size_t default_stack_size = 0;
  struct rlimit stack_rlimit;
  if (pthread_attr_getstacksize(&attributes, &default_stack_size) == 0 &&
      getrlimit(RLIMIT_STACK, &stack_rlimit) == 0 &&
      stack_rlimit.rlim_cur != RLIM_INFINITY) {
    default_stack_size =
        std::max(std::max(default_stack_size,
                          static_cast<size_t>(PTHREAD_STACK_MIN)),
                 static_cast<size_t>(stack_rlimit.rlim_cur));
  }
  return default_stack_size;
#endif
}

void TerminateOnThread() {
}

}  // namespace tachyon::base
