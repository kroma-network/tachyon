FROM --platform=linux/amd64 ubuntu:latest AS cacher

EXPOSE 80

RUN apt-get update && apt-get install nginx-extras -y && \
    rm -rf /var/lib/apt/lists/*

COPY nginx.conf /etc/nginx/nginx.conf

CMD nginx -g "daemon off;"

FROM --platform=linux/amd64 ubuntu:latest AS builder
LABEL maintainer="TA <ta@lightscale.io>"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install wget curl unzip zip build-essential git make libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev python-is-python3 -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN curl -LO https://github.com/bazelbuild/bazel/releases/download/6.2.1/bazel-6.2.1-installer-linux-x86_64.sh && \
    chmod +x bazel-6.2.1-installer-linux-x86_64.sh && \
    ./bazel-6.2.1-installer-linux-x86_64.sh --user
ENV PATH="/root/bin:${PATH}"

ENV HOME /root
RUN curl https://pyenv.run | bash
RUN CONFIGURE_OPTS=--enable-shared $HOME/.pyenv/bin/pyenv install 3.10.12
RUN mkdir -p $HOME/bin
ENV PATH="${HOME}/.pyenv/shims:${HOME}/.pyenv/bin:${HOME}/bin:${PATH}"

COPY . /usr/src/tachyon
WORKDIR /usr/src/tachyon

RUN bazel build --remote_cache=http://127.0.0.0:80 -c opt --config halo2 --//:has_openmp  --//:c_shared_object //scripts/packages/debian/runtime:debian
RUN bazel build --remote_cache=http://127.0.0.0:80 -c opt --config halo2 --//:has_openmp  --//:c_shared_object //scripts/packages/debian/dev:debian

FROM --platform=linux/amd64 ubuntu:latest AS runner
WORKDIR /usr/src/tachyon
COPY --from=builder /usr/src/tachyon/bazel-bin/scripts/packages/debian/runtime/libtachyon_0.0.1_amd64.deb /usr/src/tachyon/bazel-bin/scripts/packages/debian/runtime
COPY --from=builder /usr/src/tachyon/bazel-bin/scripts/packages/debian/dev/libtachyon-dev_0.0.1_amd64.deb /usr/src/tachyon/bazel-bin/scripts/packages/debian/dev/
RUN dpkg -i bazel-bin/scripts/packages/debian/runtime/libtachyon_0.0.1_amd64.deb && \
    dpkg -i bazel-bin/scripts/packages/debian/dev/libtachyon-dev_0.0.1_amd64.deb
