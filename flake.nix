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
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        default = pkgs.mkShell {
          packages = with pkgs; [
            pandoc
            texliveFull
            (python3.withPackages (ps: with ps; with python3Packages; [
              jupyter
              ipython
              matplotlib
              numpy
              scipy
              pandas
            ]))
          ];
        };
      });
  };
}
