#if defined(TACHYON_NODE_BINDING)

#include "tachyon/node/math/math.h"

#include "tachyon/math/elliptic_curves/bls/bls12_381/g1.h"
#include "tachyon/math/elliptic_curves/bn/bn254/g1.h"
#include "tachyon/node/math/elliptic_curves/bls/bls12_381/fq.h"
#include "tachyon/node/math/elliptic_curves/bls/bls12_381/fr.h"
#include "tachyon/node/math/elliptic_curves/bls/bls12_381/g1.h"
#include "tachyon/node/math/elliptic_curves/bn/bn254/fq.h"
#include "tachyon/node/math/elliptic_curves/bn/bn254/fr.h"
#include "tachyon/node/math/elliptic_curves/bn/bn254/g1.h"

namespace tachyon::node::math {

void AddMath(NodeModule& m) {
  NodeModule bls12_381 = m.AddSubModule("bls12_381");
  bls12_381.AddFunction("init", &tachyon::math::bls12_381::G1Curve::Init);
  bls12_381::AddFq(bls12_381);
  bls12_381::AddFr(bls12_381);
  bls12_381::AddG1(bls12_381);

  NodeModule bn254 = m.AddSubModule("bn254");
  bn254.AddFunction("init", &tachyon::math::bn254::G1Curve::Init);
  bn254::AddFq(bn254);
  bn254::AddFr(bn254);
  bn254::AddG1(bn254);
}

}  // namespace tachyon::node::math

#endif  // defined(TACHYON_NODE_BINDING)
