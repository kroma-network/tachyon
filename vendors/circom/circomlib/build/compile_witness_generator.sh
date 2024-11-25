#!/usr/bin/env bash

set -e

SUPPORTED_FIELDS=("bls12381" "bn128" "goldilocks" "grumpkin" "pallas" "secq256r1" "vesta")

function compile_witness_generator() {
  CIRCOMLIB=bazel-bin/circomlib/build/compile_witness_generator.runfiles/kroma_network_circom/circomlib
  cd ${CIRCOMLIB}/build
  cp ../../../iden3_circom/code_producers/src/c_elements/common/* .
  echo "Using $2 field..."
  cp ../../../iden3_circom/code_producers/src/c_elements/$2/* .

  echo "Splitting source file into smaller pieces..."
  rm part_*
  ./split_tool $1 --output_dir=. -- -std=c++11 -I../../external/llvm-project/clang/staging/include

  echo "Compiling everything..."
  make -j
  cd -
  mv ${CIRCOMLIB}/build/witness_generator witness_generator

  echo "Success"
}

function usage() {
  echo "Usage:"
  echo "$0 [OPTION]..."
  echo ""
  echo "-h, --help  help"
  echo "--cpp,      A path to circuit .cpp file"
  echo "-f, --field A field to use. default: bn128, supported: ${SUPPORTED_FIELDS[@]}"
  echo ""
}

function main() {
  CPP=""
  FIELD="bn128"
  while [[ $# -gt 0 ]]
  do
    case $1 in
      -h | --help)
        usage
        exit 0
      ;;
      --cpp)
        CPP=$2
        shift
        shift
      ;;
      -f | --field)
        FIELD=$2
        shift
        shift
      ;;
      *)
        echo "Unknown option: $1"
        usage
        exit 1
      ;;
    esac
  done

  if [[ ${CPP} == "" ]]
  then
    echo "No cpp file specified"
    exit 1
  fi

  if [[ ! " ${SUPPORTED_FIELDS[@]} " =~ " ${FIELD} " ]]
  then
    echo "Unsupported field: $FIELD"
    exit 1
  fi

  compile_witness_generator $CPP $FIELD
}

main "$@"
