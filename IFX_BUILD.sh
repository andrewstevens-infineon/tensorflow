#!/bin/bash
set -e
source ../SETTINGS_AND_VERSIONS.sh

LOCALJOBS=HOST_CPUS*0.7
while [[ "$1" != "" && "$1" != "--" ]]
do
    case "$1" in
    "--help"|"-?") 
        echo "`basename $0`: [--no_toco]"
        exit 1
        ;;
    "--no-config")
        NOCONFIG=1
        ;;
    "--no-build")
        NOBUILD=1
        ;;
    "--no-tflite")
        NOTFLITE=1
        ;;
    "--jobs")
        LOCALJOBS="$2"
        shift
        ;;
    "--remote")
        BAZEL_REMOTE_OPTIONS=(
           --jobs="$2" --spawn_strategy=local,remote --strategy=CppCompile=remote --remote_executor=grpc://localhost:8980
        )
        ;;
    "--override-llvm")
	BAZEL_REPO_OVERRIDES=( --override_repository=llvm-project=$(readlink -f ../..)/llvm-project )
	;;
    "--verbose")
        VERBOSE=( --subcommands=true )
        ;;
    "*")
        break
        ;;
    esac
    shift
done

# Build host native and IFX riscv32 builds of TF-lite(micro)
RISCV_SETTINGS=( TARGET_ARCH=${RV_TARGET_ARCH} TARGET_ABI=${RV_TARGET_ABI} )
# These builds only here to trigger error if build fails

TFLITE_MICRO_ROOT=${TOOLSPREFIX}/tflite_u-${TFLITE_MICRO_VERSION}


if [ -n "$RD_CLUSTER_LINUX_HOST" ]
then
  BAZEL_DISTDIR_OPTIONS=( --distdir /home/aifordes.work/share/bazel-distdir )
fi

BAZEL_CMDLINE_OPTIONS=( )

if [ -n "$MINGW64_HOST" ] 
then
  BAZEL_OPTIONS=(
      "${BAZEL_DISTDIR_OPTIONS[@]}"
      "${BAZEL_REPO_OVERRIDES[@]}"
      --verbose_failures --local_cpu_resources="$LOCALJOBS"  "${VERBOSE[@]}"
      "${BAZEL_REMOTE_OPTIONS[@]}"
      --config=dbg --cxxopt=-DTF_LITE_DISABLE_X86_NEON --copt=-DTF_LITE_DISABLE_X86_NEON
      --verbose_failures=yes
  )
else
  BAZEL_OPTIONS=(
        "${BAZEL_DISTDIR_OPTIONS[@]}"
      "${BAZEL_REPO_OVERRIDES[@]}"
      --verbose_failures --local_cpu_resources="$LOCALJOBS" --config=monolithic "${VERBOSE[@]}"
      "${BAZEL_REMOTE_OPTIONS[@]}"
      --config=dbg
      --copt=-gsplit-dwarf --copt=-O1 --cxxopt=-gsplit-dwarf --cxxopt=-O1 --cxxopt=-DTF_LITE_DISABLE_X86_NEON --copt=-DTF_LITE_DISABLE_X86_NEON
      --strip=never --fission=yes --verbose_failures=yes
  )
fi
# Build a statically linked toco command-line translator
# and TF-lite(micro) libraries

# Note need matching VC or VC build tools installation to build
# MinGW is NOT (visibly) supported.
# --subcommands logs compiler commands
if [ -z "$NOCONFIG" ]
then
(
  export PYTHON_BIN_PATH=$(which python)
  if [ -n "$MINGW64_HOST" ] 
  then
      PYTHON_BIN_PATH=$(cygpath --windows "${PYTHON_BIN_PATH}" )
      EXE_SUFFIX=.exe
      export CC_OPT_FLAGS="/arch:AVX"
      # Wild patching orgy is needed... including yes in 2020
      # workarounds for pathname/command-line length limitations.
     source msys_env.sh
  else
    export TF_NEED_GCP=0
    export TF_NEED_HDFS=0
    export TF_NEED_OPENCL=0
    export TF_ENABLE_XLA=0
    export TF_NEED_VERBS=0
    export TF_CUDA_CLANG=0
    export TF_NEED_MKL=0
    export TF_DOWNLOAD_MKL=0
    export TF_NEED_AWS=0
    export TF_NEED_MPI=0
    export TF_NEED_GDR=0
    export TF_NEED_S3=0
    export TF_NEED_OPENCL_SYCL=0
    export TF_SET_ANDROID_WORKSPACE=0
    export TF_NEED_COMPUTECPP=0
    export GCC_HOST_COMPILER_PATH=$(which gcc)
    export CC_OPT_FLAGS="-march=native -Wno-sign-compare"
    export TF_SET_ANDROID_WORKSPACE=0
    export TF_NEED_KAFKA=0
    export TF_NEED_TENSORRT=0 
    export TF_DOWNLOAD_CLANG=0
  fi
#  export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')"
  export TF_NEED_ROCM=0
  export TF_NEED_CUDA=0
  export TF_OVERRIDE_EIGEN_STRONG_INLINE=1
  export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'from distutils.sysconfig import get_python_lib;print(get_python_lib())')"
  ./configure

  rm -f .bazelrc.user
  echo configured bazel build: "${BAZEL_OPTIONS[@]}"
  for o in "${BAZEL_OPTIONS[@]}"
  do
    echo build "$o" >> .bazelrc.user
  done
)
else
  echo Skipping configure...
fi

BAZEL_TARGETS=( //tensorflow/lite/toco:toco //tensorflow/compiler/mlir/lite:tf_tfl_translate )

if [ -z "$NOBUILD" ]
then
    # Some useful recipes from the past... comment out in case needed
    #  bazel fetch "${BAZEL_DISTDIR_OPTIONS[@]}" //tensorflow/compiler/mlir/lite:tf_tfl_translate
    #bazel build "${BAZEL_OPTIONS[@]}" //third_party/aws:aws || true  # EXPECTED FAILURE but needed to unpack packages
    #bazel build --local_cpu_resources="$LOCALJOBS" --config=dbg --strip=never "${VERBOSE[@]}" //tensorflow/compiler/mlir/lite:tf_tfl_translate
    #sed -e'1,$s/"+[cd]/"+g/g' -i bazel-$(basename $(pwd))/external/aws-checksums/source/intel/crc32c_sse42_asm.c  #  BUILD FAILS MISERABG:Y WITH GCC7 FFS
  (
    # Wild patching orgy is needed... including yes in 2020
    # workarounds for pathname/command-line length limitations.
    if [ -n "$MINGW64_HOST" ] 
    then
       source msys_env.sh
    fi
    # 
    echo bazel build "${BAZEL_CMDLINE_OPTIONS[@]}"  "${BAZEL_TARGETS[@]}"
    bazel build   "${BAZEL_CMDLINE_OPTIONS[@]}"  "${BAZEL_TARGETS[@]}"
    mkdir -p ${TFLITE_MICRO_ROOT}/bin  
    rm -f ${TFLITE_MICRO_ROOT}/bin/*
    cp bazel-bin/tensorflow/compiler/mlir/lite/tf_tfl_translate${EXE_SUFFIX} \
        bazel-bin/tensorflow/lite/toco/toco${EXE_SUFFIX} \
        ${TFLITE_MICRO_ROOT}/bin
  )
fi

if [ -z "$NOTFLITE" ] 
then
    make TARGET=ifx_riscv32_mcu ${RISCV_SETTINGS[@]} BUILD_TYPE=debug clean
    make BUILD_TYPE=debug clean
    make -j 4 microlite
    make -j 4 test_executables
    echo make -j 4 TARGET=ifx_riscv32_mcu ${RISCV_SETTINGS[@]} BUILD_TYPE=debug microlite
    make -j 4 TARGET=ifx_riscv32_mcu ${RISCV_SETTINGS[@]} BUILD_TYPE=debug microlite
    make -j 4 TARGET=ifx_riscv32_mcu ${RISCV_SETTINGS[@]} microlite
    echo make -j 4 BUILD_TYPE=debug test_executables
    make -j 4 BUILD_TYPE=debug microlite

    # Actual payload - installed confiured copy of tflite(u) library and makefiles
    make TARGET=ifx_riscv32_install_only ${RISCV_SETTINGS[@]} install
    
    # Clean up afterwards because bugs in downlaods from tflite(u)
    # poison VS-code bazel target discovery
    make TARGET=ifx_riscv32_install_only ${RISCV_SETTINGS[@]} clean
    make -j 4 BUILD_TYPE=debug clean
fi
