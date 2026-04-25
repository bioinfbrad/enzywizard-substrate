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

# ==============================
# Command: enzywizard-substrate
# ==============================

# brief introduction:
'''
EnzyWizard-Substrate is a command-line tool for processing small-molecule
substrates from user-provided substrate names or SMILES strings and generating
a detailed JSON report together with substrate structure files in SDF format.
It supports multiple substrate inputs in a single run.
For substrate names input, the tool automatically retrieves substrate information
through ChEBI and PubChem APIs, and retries synonym-expanded matching
to improve name-to-SMILES resolution. Based on substrate SMILES, the tool performs substrate characterization,
including molecular fingerprint generation and basic molecular descriptor calculation.
It automatically adds hydrogens to substrate, constructs possible 3D substrate
structures, minimizes the conformation energy, and saves the resulting 3D
substrate conformations as SDF files for downstream analysis.
'''

# example usage:
'''
Example command:

enzywizard-substrate -s glucose,fructose -o examples/output/

'''

# input parameters:
'''
-s, --substrate_names
Required.
Input substrate names or SMILES strings.
Multiple substrates are supported and should be separated by ','.

Examples:
  - glucose
  - CCO
  - glucose,fructose
  - glucose,CCO,lactate

If one input item is already a valid SMILES string, it will be recorded directly.
Its internal substrate name will be automatically assigned as smiles1, smiles2, etc.

-o, --output_dir
Required.
Path to the output directory for saving the JSON report and generated substrate
structure files in SDF format.

--max_synonyms
Optional.
Maximum number of substrate synonyms retried when fetching SMILES from a substrate name.

Default:
  20

A larger value may improve recall for difficult substrate names, but will increase
API requests and runtime.

--fp_radius
Optional.
Radius used for Morgan fingerprint generation.

Default:
  2

This parameter controls the topological neighborhood size considered around each atom.
Larger values capture broader local environments.

--n_bits
Optional.
Bit size of the Morgan fingerprint vector.

Default:
  512

Larger values reduce bit collisions but increase feature dimensionality.

--num_confs
Optional.
Maximum number of 3D structures to generate for each substrate.

Default:
  5

Larger values increase conformational coverage but also increase runtime.

--prune_rms
Optional.
RMS threshold used to prune highly similar conformers during 3D conformer generation.

Default:
  0.5

Smaller values keep only more distinct conformers, while larger values allow
more similar conformers to be retained.
'''

# output content:
'''
The program outputs the following files into the output directory:

1. A JSON report
   - substrate_report_{substrate_suffix}.json

   The JSON report contains:

   - "output_type"
     A string identifying the report type:
     "enzywizard_substrate"

   - "substrates"
     A list describing processed substrates.

     Each entry contains:
     - "substrate_name"
       Resolved substrate name used internally by the program.

     - "smiles"
       Final SMILES string used for downstream processing.

     - "fingerprint"
       Morgan fingerprint bit vector of the substrate.

     - "num_atoms"
       Number of atoms in the substrate.

     - "mol_weight"
       Molecular weight of the substrate.

     - "logp"
       Calculated logP value of the substrate.

     - "structures"
       A list of generated 3D substrate conformations.

       Each structure entry contains:
       - "structure_name"
         Name of the generated substrate structure.

       - "structure_energy"
         UFF energy of the generated 3D conformation.

2. Substrate structure files in SDF format
   - {structure_name}.sdf

   One SDF file is saved for each valid generated 3D conformation.

'''

# Process:
'''
This command processes the input substrates as follows:

1. Parse input substrates
   - Read the input substrate_names string.
   - Split multiple substrates by ','.
   - Determine whether each entry is a substrate name or a valid SMILES string.

2. Resolve substrate identity
   - For valid SMILES input:
       - record the SMILES directly
       - assign an internal substrate name such as smiles1, smiles2, etc.
   - For substrate name input:
       - query ChEBI for exact matches
       - attempt exact and normalized name matching
       - query PubChem for CID and SMILES
       - retrieve PubChem synonyms
       - expand synonyms and retry ChEBI matching when necessary

3. Construct 2D molecular representation
   - Convert each resolved SMILES string into an RDKit 2D molecular object.
   - Validate molecular structure consistency.

4. Calculate substrate features
   - Compute Morgan fingerprint representation.
   - Compute selected 2D molecular descriptors, including:
       - atom count
       - molecular weight
       - logP

5. Add hydrogens
   - Automatically add explicit hydrogen atoms to each valid 2D molecule
     before 3D conformer generation using RDKit.

6. Generate 3D substrate structures
   - Generate multiple candidate 3D conformers using RDKit embedding.
   - Apply RMS-based pruning to remove highly redundant conformers.

7. Minimize energy
   - Minimize each generated 3D conformer's energy using the UFF force field.
   - Compute conformer energy values for ranking.

8. Rank and organize conformers
   - Sort valid conformers by increasing energy.
   - Rename them in a consistent order such as substrate_1, substrate_2, etc.

9. Save outputs
   - Save each valid 3D substrate conformation as an individual SDF file.
   - Generate and save a JSON report summarizing resolved substrate information,
     computed features, and generated structure metadata.
'''

# dependencies:
'''
- RDKit
- requests
- urllib3
- NumPy
'''

# references:
'''
- RDKit:
  https://www.rdkit.org/

- RDKit Book:
  https://www.rdkit.org/docs/RDKit_Book.html

- PubChem PUG REST API:
  https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest

- ChEBI:
  https://www.ebi.ac.uk/chebi/
'''