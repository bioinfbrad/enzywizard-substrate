from __future__ import annotations

import argparse

from .commands.substrate import add_substrate_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enzywizard-substrate",
        description="EnzyWizard-Substrate: Process small-molecule substrates from substrate names or SMILES strings and generate a detailed JSON report together with substrate structure files in SDF format."
    )
    add_substrate_parser(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)