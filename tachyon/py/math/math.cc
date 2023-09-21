#include "tachyon/py/math/math.h"

#include "tachyon/math/elliptic_curves/bls/bls12_381/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/py/math/elliptic_curves/bls/bls12_381/fq.h"
#include "tachyon/py/math/elliptic_curves/bls/bls12_381/fr.h"
#include "tachyon/py/math/elliptic_curves/bls/bls12_381/g1.h"
#include "tachyon/py/math/elliptic_curves/bn/bn254/fq.h"
#include "tachyon/py/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/py/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::py::math {

void AddMath(py11::module& m) {
  py11::module math = m.def_submodule("math");
  py11::module bls12_381 = math.def_submodule("bls12_381");
  bls12_381.def("init", &tachyon::math::bls12_381::G1Curve::Init);
  bls12_381::AddFq(bls12_381);
  bls12_381::AddFr(bls12_381);
  bls12_381::AddG1(bls12_381);

  py11::module bn254 = math.def_submodule("bn254");
  bn254.def("init", &tachyon::math::bn254::G1Curve::Init);
  bn254::AddFq(bn254);
  bn254::AddFr(bn254);
  bn254::AddG1(bn254);
}

}  // namespace tachyon::py::math
