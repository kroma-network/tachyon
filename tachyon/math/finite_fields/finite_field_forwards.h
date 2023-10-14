// Copyright 2022 arkworks contributors
// Use of this source code is governed by a MIT/Apache-2.0 style license that
// can be found in the LICENSE-MIT.arkworks and the LICENCE-APACHE.arkworks
// file.

#ifndef TACHYON_MATH_FINITE_FIELDS_FINITE_FIELD_FORWARDS_H_
#define TACHYON_MATH_FINITE_FIELDS_FINITE_FIELD_FORWARDS_H_

namespace tachyon::math {

template <typename Config, typename SFINAE = void>
class PrimeField;

template <typename Config>
class PrimeFieldGmp;

template <typename Config>
class PrimeFieldGpu;

template <typename Config>
class PrimeFieldGpuDebug;

template <typename Config>
class Fp2;

template <typename Config>
class Fp3;

template <typename Config>
class Fp4;

template <typename Config, typename SFINAE = void>
class Fp6;

template <typename Config>
class Fp12;

}  // namespace tachyon::math

#endif  // TACHYON_MATH_FINITE_FIELDS_FINITE_FIELD_FORWARDS_H_
