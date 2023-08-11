prefix=/usr
exec_prefix=${prefix}
includedir=${prefix}/include
libdir=/usr/lib/x86_64-linux-gnu

Name: Tachyon
Description: Tachyon is a blazing fast Zero-Knowledge Proof (ZKP) library developed in C++.
URL: https://github.com/kroma-network/tachyon
Version: %{version}
Cflags: -I${includedir}
Libs: -L${libdir} -ltachyon
