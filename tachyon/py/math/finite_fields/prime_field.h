#ifndef TACHYON_PY_MATH_FINITE_FIELDS_PRIME_FIELD_H_
#define TACHYON_PY_MATH_FINITE_FIELDS_PRIME_FIELD_H_

#include <string>

#include "pybind11/operators.h"

#include "tachyon/py/base/pybind11.h"
#include "tachyon/py/math/base/big_int.h"

namespace tachyon::py::math {

template <typename PrimeField, size_t N = PrimeField::N>
void AddPrimeField(py11::module& m, const std::string& name) {
  py11::class_<PrimeField>(m, name.data())
      .def(py11::init<const tachyon::math::BigInt<N>>())
      .def_static("zero", &PrimeField::Zero)
      .def_static("one", &PrimeField::One)
      .def_static("random", &PrimeField::Random)
      .def_static("from_dec_string", &PrimeField::FromDecString)
      .def_static("from_hex_string", &PrimeField::FromHexString)
      .def("is_zero", &PrimeField::IsZero)
      .def("is_one", &PrimeField::IsOne)
      .def("to_string", &PrimeField::ToString)
      .def("to_hex_string", &PrimeField::ToHexString,
           py11::arg("pad_zero") = false)
      .def(py11::self == py11::self)
      .def(py11::self != py11::self)
      .def(py11::self < py11::self)
      .def(py11::self <= py11::self)
      .def(py11::self > py11::self)
      .def(py11::self >= py11::self)
      .def(py11::self + py11::self)
      .def(py11::self += py11::self)
      .def(py11::self - py11::self)
      // NOTE(chokobole): See https://github.com/pybind/pybind11/issues/1893
      // .def(py11::self -= py11::self)
      .def(
          "__isub__",
          [](PrimeField& lhs, const PrimeField& rhs) { return lhs -= rhs; },
          py11::is_operator())
      .def(py11::self * py11::self)
      .def(py11::self *= py11::self)
      .def(py11::self / py11::self)
      // NOTE(chokobole): See https://github.com/pybind/pybind11/issues/1893
      // .def(py11::self /= py11::self)
      .def(
          "__idiv__",
          [](PrimeField& lhs, const PrimeField& rhs) { return lhs /= rhs; },
          py11::is_operator())
      .def(-py11::self)
      .def("double", &PrimeField::Double)
      .def("double_in_place", &PrimeField::DoubleInPlace)
      .def("square", &PrimeField::Square)
      .def("square_in_place", &PrimeField::SquareInPlace)
      .def("__repr__", [name](const PrimeField& field) {
        return absl::Substitute("$0($1)", name, field.ToString());
      });
}

}  // namespace tachyon::py::math

#endif  // TACHYON_PY_MATH_FINITE_FIELDS_PRIME_FIELD_H_
