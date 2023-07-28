// Copyright 2018 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "tachyon/base/threading/platform_thread.h"

// #include "tachyon/base/task/current_thread.h"
#include "absl/base/attributes.h"

#if BUILDFLAG(IS_FUCHSIA)
#include "tachyon/base/fuchsia/scheduler.h"
#endif

namespace tachyon::base {

namespace {

ABSL_CONST_INIT thread_local ThreadType current_thread_type =
    ThreadType::kDefault;

}  // namespace

std::string ThreadTypeToString(ThreadType thread_type) {
  switch (thread_type) {
    case ThreadType::kBackground: return "Background";
    case ThreadType::kUtility: return "Utility";
    case ThreadType::kResourceEfficient: return "ResourceEfficient";
    case ThreadType::kDefault: return "Default";
    case ThreadType::kCompositing: return "Compositing";
    case ThreadType::kDisplayCritical: return "DisplayCritical";
    case ThreadType::kRealtimeAudio: return "RealtimeAudio";
  }
  NOTREACHED();
  return "";
}

std::ostream& operator<<(std::ostream& os, ThreadType thread_type) {
  return os << ThreadTypeToString(thread_type);
}

// static
void PlatformThreadBase::SetCurrentThreadType(ThreadType thread_type) {
  MessagePumpType message_pump_type = MessagePumpType::DEFAULT;
  // TODO(chokobole):
  //   if (CurrentIOThread::IsSet()) {
  //     message_pump_type = MessagePumpType::IO;
  //   }
  // #if !BUILDFLAG(IS_NACL)
  //   else if (CurrentUIThread::IsSet()) {
  //     message_pump_type = MessagePumpType::UI;
  //   }
  // #endif
  internal::SetCurrentThreadType(thread_type, message_pump_type);
}

// static
ThreadType PlatformThreadBase::GetCurrentThreadType() {
  return current_thread_type;
}

// static
std::optional<TimeDelta> PlatformThreadBase::GetThreadLeewayOverride() {
#if BUILDFLAG(IS_FUCHSIA)
  // On Fuchsia, all audio threads run with the CPU scheduling profile that uses
  // an interval of |kAudioSchedulingPeriod|. Using the default leeway may lead
  // to some tasks posted to audio threads to be executed too late (see
  // http://crbug.com/1368858).
  if (GetCurrentThreadType() == ThreadType::kRealtimeAudio)
    return kAudioSchedulingPeriod;
#endif
  return std::nullopt;
}

// static
void PlatformThreadBase::SetNameCommon(const std::string& name) {
  // TODO(chokobole):
  // ThreadIdNameManager::GetInstance()->SetName(name);
}

namespace internal {

void SetCurrentThreadType(ThreadType thread_type,
                          MessagePumpType pump_type_hint) {
  CHECK_LE(thread_type, ThreadType::kMaxValue);
  SetCurrentThreadTypeImpl(thread_type, pump_type_hint);
  current_thread_type = thread_type;
}

}  // namespace internal

}  // namespace tachyon::base
