#ifndef TACHYON_EXPORT_H_
#define TACHYON_EXPORT_H_

#if defined(TACHYON_COMPONENT_BUILD)

#if defined(_WIN32)
#ifdef TACHYON_COMPILE_LIBRARY
#define TACHYON_EXPORT __declspec(dllexport)
#else
#define TACHYON_EXPORT __declspec(dllimport)
#endif  // defined(TACHYON_COMPILE_LIBRARY)

#else
#ifdef TACHYON_COMPILE_LIBRARY
#define TACHYON_EXPORT __attribute__((visibility("default")))
#else
#define TACHYON_EXPORT
#endif  // defined(TACHYON_COMPILE_LIBRARY)
#endif  // defined(_WIN32)

#else
#define TACHYON_EXPORT
#endif  // defined(TACHYON_COMPONENT_BUILD)

#endif  // TACHYON_EXPORT_H_