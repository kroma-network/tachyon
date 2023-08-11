#!/usr/bin/env bash

set -e

function copy_and_link_lib_linux() {
  LIB=$1
  BASENAME=$(basename $LIB)

  LIBNAME=$(echo $BASENAME | awk -F. '{print $(NF - 4)}')
  MAJOR=$(echo $BASENAME | awk -F. '{print $(NF - 2)}')
  MINOR=$(echo $BASENAME | awk -F. '{print $(NF - 1)}')
  PATCH=$(echo $BASENAME | awk -F. '{print $(NF)}')

  LINK_LIB=${LIBNAME}.so
  MAJOR_LIB=${LIBNAME}.so.${MAJOR}
  FULL_LIB=${BASENAME}

  cp $LIB ${DSTDIR}/lib
  ln -sf ${DSTDIR}/lib/${FULL_LIB} ${DSTDIR}/lib/${MAJOR_LIB}
  ln -sf ${DSTDIR}/lib/${MAJOR_LIB} ${DSTDIR}/lib/${LINK_LIB}
}

function copy_and_link_lib_macos() {
  LIB=$1
  BASENAME=$(basename $LIB)

  LIBNAME=$(echo $BASENAME | awk -F. '{print $(NF - 4)}')
  MAJOR=$(echo $BASENAME | awk -F. '{print $(NF - 3)}')
  MINOR=$(echo $BASENAME | awk -F. '{print $(NF - 2)}')
  PATCH=$(echo $BASENAME | awk -F. '{print $(NF - 1)}')

  LINK_LIB=${LIBNAME}.dylib
  MAJOR_LIB=${LIBNAME}.${MAJOR}.dylib
  FULL_LIB=${BASENAME}

  cp $LIB ${DSTDIR}/lib
  ln -sf ${DSTDIR}/lib/${FULL_LIB} ${DSTDIR}/lib/${MAJOR_LIB}
  ln -sf ${DSTDIR}/lib/${MAJOR_LIB} ${DSTDIR}/lib/${LINK_LIB}
}

function install_package() {
  mkdir -p ${DSTDIR}/include ${DSTDIR}/lib
  RUNFILES=bazel-bin/scripts/packages/install_package.runfiles/kroma_network_tachyon

  FILES=`find ${RUNFILES}/tachyon -type l`
  echo "Copying libs to ${DSTDIR}/lib"
  for FILE in $FILES
  do
    if [[ "$FILE" =~ .*so.[[:digit:]].[[:digit:]].[[:digit:]] ]]; then
      copy_and_link_lib_linux $FILE
    elif [[ "$FILE" =~ .*.[[:digit:]].[[:digit:]].[[:digit:]].dylib ]]; then
      copy_and_link_lib_macos $FILE
    fi
  done
  echo "Done copying libs to ${DSTDIR}/lib"

  FILES=`find ${RUNFILES}/tachyon -name "*.h"`
  echo "Copying headers to ${DSTDIR}/include"
  for FILE in $FILES
  do
    ${RUNFILES}/scripts/packages/copy_hdr.py ${FILE} ${DSTDIR}/include --strip ${RUNFILES}/
  done
  echo "Done copying headers to ${DSTDIR}/include"
}

function usage() {
  echo "Usage:"
  echo "$0 [OPTION]..."
  echo ""
  echo "-h, --help                      help"
  echo "-d, --dst                       absolute path to install. default: /opt/kroma/tachyon"
  echo ""
}

function main() {
  DSTDIR="/opt/kroma/tachyon"
  while [[ $# -gt 0 ]]
  do
  case $1 in
    -h | --help)
    usage
    exit 0
    ;;
    -d | --dst)
    DSTDIR="$2"
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

  install_package
}

main "$@"
