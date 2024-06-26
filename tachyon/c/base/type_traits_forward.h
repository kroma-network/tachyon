#ifndef TACHYON_C_BASE_TYPE_TRAITS_FORWARD_H_
#define TACHYON_C_BASE_TYPE_TRAITS_FORWARD_H_

namespace tachyon::c::base {

template <typename T>
struct TypeTraits;

template <typename CType,
          typename NativeType = typename TypeTraits<CType>::NativeType>
const NativeType& native_cast(const CType& c_type) {
  return reinterpret_cast<const NativeType&>(c_type);
}

template <typename CType,
          typename NativeType = typename TypeTraits<CType>::NativeType>
NativeType& native_cast(CType& c_type) {
  return reinterpret_cast<NativeType&>(c_type);
}

template <typename CType,
          typename NativeType = typename TypeTraits<CType>::NativeType>
NativeType&& native_cast(CType&& c_type) {
  return reinterpret_cast<NativeType&&>(c_type);
}

template <typename CType,
          typename NativeType = typename TypeTraits<CType>::NativeType>
const NativeType* native_cast(const CType* c_type) {
  return reinterpret_cast<const NativeType*>(c_type);
}

template <typename CType,
          typename NativeType = typename TypeTraits<CType>::NativeType>
NativeType* native_cast(CType* c_type) {
  return reinterpret_cast<NativeType*>(c_type);
}

template <typename NativeType,
          typename CType = typename TypeTraits<NativeType>::CType>
const CType& c_cast(const NativeType& native_type) {
  return reinterpret_cast<const CType&>(native_type);
}

template <typename NativeType,
          typename CType = typename TypeTraits<NativeType>::CType>
CType& c_cast(NativeType& native_type) {
  return reinterpret_cast<CType&>(native_type);
}

template <typename NativeType,
          typename CType = typename TypeTraits<NativeType>::CType>
CType&& c_cast(NativeType&& native_type) {
  return reinterpret_cast<CType&&>(native_type);
}

template <typename NativeType,
          typename CType = typename TypeTraits<NativeType>::CType>
const CType* c_cast(const NativeType* native_type) {
  return reinterpret_cast<const CType*>(native_type);
}

template <typename NativeType,
          typename CType = typename TypeTraits<NativeType>::CType>
CType* c_cast(NativeType* native_type) {
  return reinterpret_cast<CType*>(native_type);
}

}  // namespace tachyon::c::base

#endif  // TACHYON_C_BASE_TYPE_TRAITS_FORWARD_H_
