// Copyright 2019 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// clang-format off

#ifndef TACHYON_BASE_IMMEDIATE_CRASH_H_
#define TACHYON_BASE_IMMEDIATE_CRASH_H_

#include "tachyon/build/build_config.h"

// Crashes in the fastest possible way with no attempt at logging.
// There are several constraints; see http://crbug.com/664209 for more context.
//
// - TRAP_SEQUENCE_() must be fatal. It should not be possible to ignore the
//   resulting exception or simply hit 'continue' to skip over it in a debugger.
// - Different instances of TRAP_SEQUENCE_() must not be folded together, to
//   ensure crash reports are debuggable. Unlike __builtin_trap(), asm volatile
//   blocks will not be folded together.
//   Note: TRAP_SEQUENCE_() previously required an instruction with a unique
//   nonce since unlike clang, GCC folds together identical asm volatile
//   blocks.
// - TRAP_SEQUENCE_() must produce a signal that is distinct from an invalid
//   memory access.
// - TRAP_SEQUENCE_() must be treated as a set of noreturn instructions.
//   __builtin_unreachable() is used to provide that hint here. clang also uses
//   this as a heuristic to pack the instructions in the function epilogue to
//   improve code density.
// - base::ImmediateCrash() is used in allocation hooks. To prevent recursions,
//   TRAP_SEQUENCE_() must not allocate.
//
// Additional properties that are nice to have:
// - TRAP_SEQUENCE_() should be as compact as possible.
// - The first instruction of TRAP_SEQUENCE_() should not change, to avoid
//   shifting crash reporting clusters. As a consequence of this, explicit
//   assembly is preferred over intrinsics.
//   Note: this last bullet point may no longer be true, and may be removed in
//   the future.

// Note: TRAP_SEQUENCE Is currently split into two macro helpers due to the fact
// that clang emits an actual instruction for __builtin_unreachable() on certain
// platforms (see https://crbug.com/958675). In addition, the int3/bkpt/brk will
// be removed in followups, so splitting it up like this now makes it easy to
// land the followups.

#if defined(COMPILER_GCC)

#if BUILDFLAG(IS_NACL)

// Crash report accuracy is not guaranteed on NaCl.
#define TRAP_SEQUENCE1_() __builtin_trap()
#define TRAP_SEQUENCE2_() asm volatile("")

#elif defined(ARCH_CPU_X86_FAMILY)

// TODO(https://crbug.com/958675): In theory, it should be possible to use just
// int3. However, there are a number of crashes with SIGILL as the exception
// code, so it seems likely that there's a signal handler that allows execution
// to continue after SIGTRAP.
#define TRAP_SEQUENCE1_() asm volatile("int3")

#if BUILDFLAG(IS_APPLE)
// Intentionally empty: __builtin_unreachable() is always part of the sequence
// (see IMMEDIATE_CRASH below) and already emits a ud2 on Mac.
#define TRAP_SEQUENCE2_() asm volatile("")
#else
#define TRAP_SEQUENCE2_() asm volatile("ud2")
#endif  // BUILDFLAG(IS_APPLE)

#elif defined(ARCH_CPU_ARMEL)

// bkpt will generate a SIGBUS when running on armv7 and a SIGTRAP when running
// as a 32 bit userspace app on arm64. There doesn't seem to be any way to
// cause a SIGTRAP from userspace without using a syscall (which would be a
// problem for sandboxing).
// TODO(https://crbug.com/958675): Remove bkpt from this sequence.
#define TRAP_SEQUENCE1_() asm volatile("bkpt #0")
#define TRAP_SEQUENCE2_() asm volatile("udf #0")

#elif defined(ARCH_CPU_ARM64)

// This will always generate a SIGTRAP on arm64.
// TODO(https://crbug.com/958675): Remove brk from this sequence.
#define TRAP_SEQUENCE1_() asm volatile("brk #0")
#define TRAP_SEQUENCE2_() asm volatile("hlt #0")

#else

// Crash report accuracy will not be guaranteed on other architectures, but at
// least this will crash as expected.
#define TRAP_SEQUENCE1_() __builtin_trap()
#define TRAP_SEQUENCE2_() asm volatile("")

#endif  // ARCH_CPU_*

#elif defined(COMPILER_MSVC)

#if !defined(__clang__)

// MSVC x64 doesn't support inline asm, so use the MSVC intrinsic.
#define TRAP_SEQUENCE1_() __debugbreak()
#define TRAP_SEQUENCE2_()

#elif defined(ARCH_CPU_ARM64)

// Windows ARM64 uses "BRK #F000" as its breakpoint instruction, and
// __debugbreak() generates that in both VC++ and clang.
#define TRAP_SEQUENCE1_() __debugbreak()
// Intentionally empty: __builtin_unreachable() is always part of the sequence
// (see IMMEDIATE_CRASH below) and already emits a ud2 on Win64,
// https://crbug.com/958373
#define TRAP_SEQUENCE2_() __asm volatile("")

#else

#define TRAP_SEQUENCE1_() asm volatile("int3")
#define TRAP_SEQUENCE2_() asm volatile("ud2")

#endif  // __clang__

#else

#error No supported trap sequence!

#endif  // COMPILER_GCC

#define TRAP_SEQUENCE_() \
  do {                   \
    TRAP_SEQUENCE1_();   \
    TRAP_SEQUENCE2_();   \
  } while (false)

// This version of ALWAYS_INLINE inlines even in is_debug=true.
// TODO(pbos): See if NDEBUG can be dropped from ALWAYS_INLINE as well, and if
// so merge. Otherwise document why it cannot inline in debug in
// base/compiler_specific.h.
#if defined(COMPILER_GCC)
#define IMMEDIATE_CRASH_ALWAYS_INLINE inline __attribute__((__always_inline__))
#elif defined(COMPILER_MSVC)
#define IMMEDIATE_CRASH_ALWAYS_INLINE __forceinline
#else
#define IMMEDIATE_CRASH_ALWAYS_INLINE inline
#endif

namespace tachyon::base {

[[noreturn]] IMMEDIATE_CRASH_ALWAYS_INLINE void ImmediateCrash() {
  TRAP_SEQUENCE_();
#if defined(__clang__) || defined(COMPILER_GCC)
  __builtin_unreachable();
#endif  // defined(__clang__) || defined(COMPILER_GCC)
}

}  // namespace tachyon::base

#endif  // TACHYON_BASE_IMMEDIATE_CRASH_H_

// clang-format on
