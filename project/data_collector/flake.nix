{
  description = "Basic template";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" ];
      forEachSystem = f: nixpkgs.lib.genAttrs  systems (system: f system);
    in
  {
    devShells = forEachSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowInsecurePackages = true;
            permittedInsecurePackages = [
              "python-2.7.18.8"
            ];
          };
        };
      in {
        default = pkgs.mkShell {
          packages = with pkgs; [
            python27
            python27Packages.numpy
            python27Packages.pandas           
          ];
        };
      });
  };
}
