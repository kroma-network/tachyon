#ifndef TACHYON_ZK_PLONK_HALO2_PROVING_SCHEME_H_
#define TACHYON_ZK_PLONK_HALO2_PROVING_SCHEME_H_

#include "tachyon/zk/lookup/type.h"
#include "tachyon/zk/plonk/halo2/vendor.h"

namespace tachyon::zk::plonk::halo2 {

template <Vendor _Vendor, lookup::Type _LookupType, typename _PCS>
class ProvingScheme {
 public:
  constexpr static Vendor kVendor = _Vendor;
  constexpr static lookup::Type kLookupType = _LookupType;

  using PCS = _PCS;
};

}  // namespace tachyon::zk::plonk::halo2

#endif  // TACHYON_ZK_PLONK_HALO2_PROVING_SCHEME_H_
