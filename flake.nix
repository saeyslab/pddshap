{
  description = "Development environment for pddshap";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Manylinux wheels (numpy, pandas, h5py, matplotlib, numba, ...) expect to
        # dynamically link against shared libraries that don't live on NixOS's
        # default library search path. Expose them via LD_LIBRARY_PATH so `uv sync`
        # installed wheels work without recompiling from source.
        libPath = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
          pkgs.libGL
          pkgs.glib
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.just
            pkgs.uv
          ];

          LD_LIBRARY_PATH = libPath;

          shellHook = ''
            uv sync
            source .venv/bin/activate
          '';
        };
      }
    );
}
