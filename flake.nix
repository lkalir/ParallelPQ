{
  description = "ParallelPQ dev shell: CUDA, ROCm/HIP, CMake, Google Benchmark";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  nixConfig = {
    extra-substituters = [ "https://cuda-maintainers.cachix.org" ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      cp = pkgs.cudaPackages;
      rp = pkgs.rocmPackages;
      cudaEnv = pkgs.buildEnv {
        name = "cuda-env-${cp.cudaMajorMinorVersion}";
        paths = with cp; [
          cuda_nvcc
          cuda_cudart
          cccl
          nsight_systems
        ];
        extraOutputsToInstall = [
          "dev"
          "include"
          "lib"
        ];
        pathsToLink = [
          "/bin"
          "/include"
          "/lib"
          "/nvvm"
        ];
        ignoreCollisions = true;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          # Tooling
          cmake
          ninja
          pkg-config
          gdb
          strace
          clang-tools
          gcc
          clang
          clang-analyzer
          cmake-language-server
          gbenchmark
          python3
          python3Packages.matplotlib
          python3Packages.ruff
          python3Packages.ty
          python3Packages.grip

          # CUDA
          cp.cuda_nvcc
          cp.cuda_cudart

          # ROCm + HIP
          rp.clr
          rp.hipcc
          rp.rocm-runtime
          rp.rocm-cmake
          rp.rocminfo
          rp.rocm-smi
        ];

        shellHook = ''
          export CUDA_PATH=${cudaEnv}
          export CUDA_HOME=$CUDA_PATH
          export ROCM_PATH=${rp.clr}
          export HIP_PATH=${rp.clr}
          export NIX_ENFORCE_NO_NATIVE=0

          has_amd=0;
          has_nvidia=0;
          wsl_lib="";
          [ -e /dev/kfd ] && has_amd=1

          if [ -e /proc/driver/nvidia/version ] || [ -d /sys/module/nvidia ]; then
            has_nvidia=1
          fi

          if [ -e /dev/dxg ] && [ -d /usr/lib/wsl/lib ]; then
            wsl_lib=/usr/lib/wsl/lib
            ls $wsl_lib/libcuda.so* >/dev/null 2>&1 && has_nvidia=1
          fi

          libpath="${rp.clr}/lib"
          [ -n "$wsl_lib" ] && libpath="$libpath:$wsl_lib"
          [ "$has_nvidia" = 1 ] && libpath="$libpath:/run/opengl-driver/lib"
          export LD_LIBRARY_PATH="$libpath''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

          if [ "$has_amd" = 1 ]; then
            export HIP_PLATFORM=amd
          elif [ "$has_nvidia" = 1 ]; then
            export HIP_PLATFORM=nvidia
            export HIP_COMPILER=nvcc
            export PATH="${cp.nsight_systems}/bin:$PATH"
          fi
        '';
      };
    };
}
