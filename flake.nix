{
  description = "Neuromorphic Intermediate Representation PyTorch graph analysis layer"; 
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system}; 
          py = pkgs.python3Packages;
      in { 
        devShells.default = pkgs.mkShell rec {
            name = "impurePythonEnv";
            venvDir = "./.venv";
            buildInputs = [
              pkgs.stdenv.cc.cc.lib
              py.python
              py.venvShellHook
              py.numpy
              py.h5py
              py.black
              py.torch
              pkgs.ruff
            ];
            postVenvCreation = ''
              unset SOURCE_DATE_EPOCH
            '';
            postShellHook = ''
              unset SOURCE_DATE_EPOCH
              export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc
              ]}
              pip install -e .
            '';
          };
      }
    );
}
