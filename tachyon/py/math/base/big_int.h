#ifndef TACHYON_PY_MATH_BASE_BIG_INT_H_
#define TACHYON_PY_MATH_BASE_BIG_INT_H_

#include "pybind11/pybind11.h"

#include "tachyon/build/build_config.h"
#include "tachyon/math/base/big_int.h"

namespace pybind11 {
namespace detail {

inline constexpr const char* GetByteOrder() {
#if ARCH_CPU_BIG_ENDIAN == 1
  return "big";
#else
  return "little";
#endif
}

#define DEFINE_BIGINT_TYPE_CASTER(N)                                          \
  template <>                                                                 \
  struct type_caster<tachyon::math::BigInt<N>> {                              \
   public:                                                                    \
    PYBIND11_TYPE_CASTER(tachyon::math::BigInt<N>,                            \
                         const_name("tachyon.math.BigInt" #N));               \
                                                                              \
    bool load(handle src, bool) {                                             \
      PyObject* source = src.ptr();                                           \
      PyObject* bit_length_obj =                                              \
          PyObject_CallMethod(source, "bit_length", nullptr);                 \
      size_t bit_length = PyLong_AsSize_t(bit_length_obj);                    \
      Py_DECREF(bit_length_obj);                                              \
      size_t byte_length = (bit_length + 7) / 8;                              \
      if (byte_length > N) return false;                                      \
      PyObject* bytes =                                                       \
          PyObject_CallMethod(source, "to_bytes", "(is)",                     \
                              static_cast<int>(byte_length), GetByteOrder()); \
      if (!bytes) return false;                                               \
      char* buf;                                                              \
      Py_ssize_t len;                                                         \
      int ret = PyBytes_AsStringAndSize(bytes, &buf, &len);                   \
      Py_DECREF(bytes);                                                       \
      if (ret == -1) return false;                                            \
      memcpy(&value.limbs[0], buf, len);                                      \
      return !PyErr_Occurred();                                               \
    }                                                                         \
                                                                              \
    static handle cast(const tachyon::math::BigInt<N>& src,                   \
                       return_value_policy, handle) {                         \
      return PyBytes_FromStringAndSize(                                       \
          reinterpret_cast<const char*>(src.limbs), N);                       \
    }                                                                         \
  }

DEFINE_BIGINT_TYPE_CASTER(4);
DEFINE_BIGINT_TYPE_CASTER(6);

}  // namespace detail
}  // namespace pybind11

#endif  // TACHYON_PY_MATH_BASE_BIG_INT_H_
