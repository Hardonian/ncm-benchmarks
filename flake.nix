{
  description = "Reproducible offline devShell for ncm-benchmarks";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forAll = nixpkgs.lib.genAttrs systems;
    in {
      devShells = forAll (system:
        let pkgs = import nixpkgs { inherit system; };
        in {
          default = pkgs.mkShell {
            packages = [ pkgs.python312 pkgs.ruff pkgs.git ];
            shellHook = ''
              export HF_HUB_OFFLINE=1
              export HF_HUB_DISABLE_XET=1
            '';
          };
        });
    };
}
