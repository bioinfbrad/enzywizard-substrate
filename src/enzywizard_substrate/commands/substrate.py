from __future__ import annotations
from argparse import Namespace, ArgumentParser
from ..services.substrate_service import run_substrate_service


def add_substrate_parser(parser: ArgumentParser) -> None:
    parser.add_argument("-s","--substrate_names",required=True,help="Input substrate names or SMILES strings. Multiple substrate names/SMILES should be separated by ','.")
    parser.add_argument("-o","--output_dir",required=True,help="Path to the output directory for saving the JSON report and generated substrate structure files in SDF format.")
    parser.add_argument("--max_synonyms",type=int,default=20,help="Maximum number of substrate synonyms retried when fetching SMILES from a substrate name (default: 20). A larger value may improve recall but will increase API requests and runtime.")
    parser.add_argument("--fp_radius",type=int,default=2,help="Radius used for Morgan fingerprint generation (default: 2). This controls the topological neighborhood size considered around each atom. Larger values capture broader local environments but may produce sparser fingerprints.")
    parser.add_argument("--n_bits",type=int,default=512,help="Bit size of the Morgan fingerprint vector (default: 512). Larger values reduce bit collisions but increase feature dimensionality.")
    parser.add_argument("--num_confs",type=int,default=5,help="Maximum number of 3D structures to generate for each substrate (default: 5). Larger values increase conformational coverage but also increase runtime.")
    parser.add_argument("--prune_rms",type=float,default=0.5,help="RMS threshold used to prune highly similar conformers during 3D conformer generation (default: 0.5). Smaller values keep more distinct conformers only, while larger values allow more similar conformers to be retained.")

    parser.set_defaults(func=run_substrate)


def run_substrate(args: Namespace) -> None:
    run_substrate_service(
        substrate_names=args.substrate_names,
        output_dir=args.output_dir,
        max_synonyms=args.max_synonyms,
        fp_radius=args.fp_radius,
        n_bits=args.n_bits,
        num_confs=args.num_confs,
        prune_rms=args.prune_rms,
    )

