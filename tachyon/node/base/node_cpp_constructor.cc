#include "tachyon/node/base/node_cpp_constructor.h"

#include "tachyon/base/strings/string_util.h"
#include "tachyon/node/base/node_constructors.h"
#include "tachyon/node/base/node_errors.h"

namespace tachyon::node {

NodeCppConstructor::NodeCppConstructor() = default;

NodeCppConstructor::~NodeCppConstructor() = default;

bool NodeCppConstructor::Match(const Napi::CallbackInfo& info) const {
  size_t nargs = info.Length();
  if (arg_type_names_.size() - default_args_num_ > nargs ||
      nargs > arg_type_names_.size())
    return false;

  for (size_t i = 0; i < nargs; ++i) {
    Napi::Value value = info[i];
    std::string_view arg_type_name = arg_type_names_[i];
    napi_valuetype type = value.Type();
    switch (type) {
      case napi_undefined:
        return false;
      case napi_null:
        return false;
      case napi_boolean:
        if (arg_type_name != base::kCppBoolTypeName) return false;
        break;
      case napi_number:
        if (arg_type_name != base::kCppNumberTypeName) return false;
        break;
      case napi_string:
        if (arg_type_name != base::kCppStringTypeName) return false;
        break;
      case napi_bigint:
        if (arg_type_name != base::kCppInt64TypeName &&
            arg_type_name != base::kCppUint64TypeName)
          return false;
        break;
      case napi_object: {
        if (value.IsArray()) {
          if (!base::StartsWith(arg_type_name, base::kCppVectorTypePrefix))
            return false;
        } else if (value.IsArrayBuffer()) {
        } else if (value.IsTypedArray()) {
        } else if (value.IsPromise()) {
        } else if (value.IsBuffer()) {
        } else if (value.IsFunction()) {
        } else if (value.IsDataView()) {
        } else {
          std::string_view name = UnwrapPointer(arg_type_name);
          Napi::Object object = value.ToObject();
          Napi::Value is_const = object.Get("__is_const__");
          CHECK(is_const.IsBoolean());
          if (!base::StartsWith(name, "const ") &&
              is_const.As<Napi::Boolean>().Value()) {
            return false;
          }

          if (!NodeConstructors::GetInstance().InstanceOf(object, name))
            return false;
        }
        break;
      }
      default:
        NOTIMPLEMENTED() << "Unknown js type: " << type;
        return false;
    }
  }
  return true;
}

}  // namespace tachyon::node
