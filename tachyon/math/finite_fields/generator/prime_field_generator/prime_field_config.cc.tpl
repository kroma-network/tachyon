// clang-format off
namespace %{namespace} {

%{if kHasTwoAdicRootOfUnity}
// static
BigInt<%{n}> %{class}Config::kSubgroupGenerator;
// static
BigInt<%{n}> %{class}Config::kTwoAdicRootOfUnity;
%{if kHasLargeSubgroupRootOfUnity}
// static
BigInt<%{n}> %{class}Config::kLargeSubgroupRootOfUnity;
%{endif kHasLargeSubgroupRootOfUnity}
%{endif kHasTwoAdicRootOfUnity}

}  // namespace %{namespace}
// clang-format on
