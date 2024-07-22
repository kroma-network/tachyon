// clang-format off
namespace %{namespace} {

%{if kHasTwoAdicRootOfUnity}
// static
uint32_t %{class}Config::kSubgroupGenerator;
// static
uint32_t %{class}Config::kTwoAdicRootOfUnity;
%{endif kHasTwoAdicRootOfUnity}

}  // namespace %{namespace}
// clang-format on
