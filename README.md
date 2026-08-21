[![DOI](https://zenodo.org/badge/1219037701.svg)](https://doi.org/10.5281/zenodo.19710033)
# EnzyWizard-Substrate

EnzyWizard-Substrate is a command-line tool for processing small-molecule
substrates from user-provided substrate names or SMILES strings and generating
a detailed JSON report together with substrate structure files in SDF format.
It supports multiple substrate inputs in a single run.
For substrate names input, the tool automatically retrieves substrate information
through ChEBI and PubChem APIs, and retries synonym-expanded matching
to improve name-to-SMILES resolution. Based on substrate SMILES, the tool performs substrate characterization,
including molecular fingerprint encoding generation and basic molecular descriptor calculation.
It automatically adds hydrogens to substrate, constructs possible molecular structures,
minimizes the conformation energy, and saves the resulting 3D
substrate conformations as SDF files for downstream analysis.


# Documentation index:

- example usage
- input parameters
- output files
- output report schema
- Process
- common errors and solutions
- dependencies
- references


# example usage:

The examples below use placeholder paths such as `path/to/output_dir/`;
replace them with your own output directory. EnzyWizard-Substrate does not take
an input structure file. Instead, substrates are provided directly as names or
SMILES strings with `-s`. Name inputs are resolved to SMILES through the ChEBI
and PubChem APIs, while direct SMILES inputs skip external lookup. Generated 3D
structures are written as SDF files.

Generate a substrate report and SDF structures from a single substrate name with
default settings.

```
enzywizard-substrate -s "glucose" -o path/to/output_single_name/
```

Generate a substrate report and SDF structures from multiple substrate names.
Multiple inputs are separated by semicolons.

```
enzywizard-substrate -s "glucose;fructose" -o path/to/output_dir/
```

Generate substrate features from a single SMILES string. Direct SMILES input
skips external name-to-SMILES lookup and is useful when the exact chemical
structure is already known.

```
enzywizard-substrate -s "CCO" -o path/to/output_smiles/
```

Run a mixed input containing a substrate name and a SMILES string. SMILES entries
are assigned internal names such as `smiles1`, while named substrates keep their
resolved names.

```
enzywizard-substrate -s "glucose;CCO" -o path/to/output_mixed/
```

Use long option names for the same basic workflow.

```
enzywizard-substrate --substrate_names "glucose;fructose" --output_dir path/to/output_dir/
```

Use fewer PubChem synonyms when resolving substrate names. This can reduce API
requests and runtime, but it may miss difficult or ambiguous substrate names
that require synonym-expanded matching.

```
enzywizard-substrate -s "glucose;fructose" -o path/to/output_fast_lookup/ --max_synonyms 5
```

Use more PubChem synonyms when resolving substrate names. This may improve
recall for difficult names, but it increases API requests and runtime.

```
enzywizard-substrate -s "glucose;fructose" -o path/to/output_broad_lookup/ --max_synonyms 100
```

Use a smaller Morgan fingerprint radius and bit vector. This produces a more
compact fingerprint and can be faster, but it captures less extended local
chemical environment and may increase bit collisions when `--n_bits` is small.

```
enzywizard-substrate -s "glucose;fructose" -o path/to/output_compact_fp/ --fp_radius 1 --n_bits 256
```

Use a larger Morgan fingerprint radius and bit vector. This captures broader
topological neighborhoods and reduces bit collisions, but it increases feature
dimensionality.

```
enzywizard-substrate -s "glucose;fructose" -o path/to/output_large_fp/ --fp_radius 3 --n_bits 1024
```

Generate fewer 3D conformers for a faster run. This reduces conformational
coverage and may miss alternative low-energy structures.

```
enzywizard-substrate -s "glucose;fructose" -o path/to/output_few_confs/ --num_confs 2
```

Generate more 3D conformers for broader conformational coverage. This may find
more candidate structures, but it increases runtime.

```
enzywizard-substrate -s "glucose;fructose" -o path/to/output_more_confs/ --num_confs 10
```

Use a smaller RMS pruning threshold to prune less aggressively. This can retain
more similar conformers and broader raw conformer coverage, but may include more
redundant conformations.

```
enzywizard-substrate -s "glucose;fructose" -o path/to/output_distinct_confs/ --prune_rms 0.2
```

Use a larger RMS pruning threshold to prune more aggressively. This keeps only
more distinct conformers and can reduce redundancy, but may reduce the number of
reported candidate conformations.

```
enzywizard-substrate -s "glucose;fructose" -o path/to/output_dense_confs/ --prune_rms 1.0
```

Demonstrate a combined parameter setting with lookup, fingerprint, and conformer
generation options in one command.

```
enzywizard-substrate -s "glucose;fructose" -o path/to/output_screening/ --max_synonyms 50 --fp_radius 3 --n_bits 1024 --num_confs 10 --prune_rms 0.3
```



# input parameters:

-s, --substrate_names
Required.
Input substrate names or SMILES strings.
Multiple substrates are supported and should be separated by ';'.
Substrate names are resolved to SMILES through ChEBI and PubChem APIs.

Examples:
  - glucose
  - CCO
  - glucose;fructose
  - glucose;CCO;lactate

If one input item is already a valid SMILES string, it will be recorded directly.
Its internal substrate name will be automatically assigned as smiles1, smiles2, etc.

-o, --output_dir
Required.
Path to the output directory for saving the JSON report and generated substrate
structure files in SDF format.
The output directory is created automatically if it does not exist.

--max_synonyms
Optional.
Maximum number of substrate synonyms retried when fetching SMILES from a substrate name.
Default: 20.
Must be between 1 and 200.
A smaller value reduces API requests and runtime, but may miss difficult or
ambiguous substrate names. A larger value may improve name-to-SMILES recall, but
increases API requests and runtime.

--fp_radius
Optional.
Radius used for Morgan fingerprint encoding generation.
Default: 2.
Must be between 1 and 5.
This parameter controls the topological neighborhood size considered around each atom.
Smaller values create a more local fingerprint. Larger values capture broader
local chemical environments and may change the resulting fingerprint encoding.

--n_bits
Optional.
Bit size of the Morgan fingerprint encoding vector.
Default: 512.
Must be between 1 and 2048.
Smaller values produce a more compact fingerprint vector, but can increase bit
collisions. Larger values reduce bit collisions, but increase feature
dimensionality.

--num_confs
Optional.
Maximum number of possible molecular structures to generate for each substrate.
Default: 5.
Must be between 1 and 20.
Smaller values run faster, but reduce conformational coverage and may miss
alternative low-energy structures. Larger values increase conformational
coverage, but also increase runtime.

--prune_rms
Optional.
RMS threshold used to prune highly similar conformers during 3D conformer generation.
Default: 0.5.
Must be greater than 0 and less than or equal to 5.0.
Conformers closer than this RMS threshold to an already accepted conformer are
pruned. Smaller values prune less aggressively and may retain more similar
conformers. Larger values prune more aggressively and keep only more distinct
conformers.


# output files:

The program outputs the following files into the output directory:

In `{substrate_name}`, `substrate_name` is the resolved substrate name such as `glucose`. For
multiple substrates, the substrate names correspond to the input order, for
example `glucose` and `fructose` for input `glucose;fructose`. For direct SMILES
inputs, the automatically assigned names are used, such as `smiles1`, `smiles2`,
etc.

1. A JSON report
   - substrate_report_{substrate_name}.json
     - JSON report containing substrate features and generated substrate conformer details.
     - For multiple substrates, the report file name uses the resolved substrate
       names in input order, joined with underscores, for example
       `substrate_report_glucose_fructose.json` or `substrate_report_smiles1_smiles2.json`.

2. Substrate structure files in SDF format
   - {substrate_name}_{index}.sdf
     - Generated 3D substrate conformation in SDF format. `{index}` is the
       generated conformer index for that substrate.

3. A log file
   - log.txt
     - Processing log containing informational messages and errors.


# output report schema:

The JSON report contains the following fields:

- "report_type"
  - Data type: string
  - Expected value: "enzywizard_substrate"
  - Description: The field 'report_type' indicates the type of report ('report': http://purl.obolibrary.org/obo/IAO_0000088) generated by the EnzyWizard-Substrate software.

- "substrates"
  - Data type: array
  - Description: The field 'substrates' indicates the processed substrates ('substrate': https://purl.dsmz.de/schema/Substrate).

  Each item in "substrates" is an object containing:

  - "substrate_name"
    - Data type: string
    - Description: The field 'substrate_name' indicates the name of the substrate ('substrate': https://purl.dsmz.de/schema/Substrate).

  - "substrate_smiles"
    - Data type: string
    - Description: The field 'substrate_smiles' indicates the SMILES representation ('SMILES': https://opensmiles.org/opensmiles.html) of the substrate ('substrate': https://purl.dsmz.de/schema/Substrate).

  - "substrate_fingerprint_encoding"
    - Data type: array
    - Item data type: integer
    - Item allowed values: 0, 1
    - Description: The field 'substrate_fingerprint_encoding' indicates the molecular fingerprint encoding ('molecular fingerprint': https://www.rdkit.org/docs/GettingStartedInPython.html#fingerprinting-and-molecular-similarity) of the substrate ('substrate': https://purl.dsmz.de/schema/Substrate) calculated by RDKit software ('RDKit': https://www.rdkit.org/docs/index.html).

  - "substrate_atom_count"
    - Data type: integer
    - Description: The field 'substrate_atom_count' indicates the count of atoms ('atom': https://goldbook.iupac.org/terms/view/A00493) in the substrate ('substrate': https://purl.dsmz.de/schema/Substrate).

  - "substrate_molecular_weight"
    - Data type: number
    - Description: The field 'substrate_molecular_weight' indicates the molecular weight ('molecular weight': https://goldbook.iupac.org/terms/view/R05271) of the substrate ('substrate': https://purl.dsmz.de/schema/Substrate). Unit: daltons (Da) ('dalton': http://qudt.org/vocab/unit/DA).

  - "substrate_logp"
    - Data type: number
    - Description: The field 'substrate_logp' indicates the calculated logP value ('LogP': https://doktormike.gitlab.io/posts/navigating-logp-logd-pka-and-logs-a-physicists-guide/) of the substrate ('substrate': https://purl.dsmz.de/schema/Substrate). Unit: dimensionless ('dimensionless': http://qudt.org/vocab/unit/UNITLESS).

  - "substrate_tpsa"
    - Data type: number
    - Description: The field 'substrate_tpsa' indicates the topological polar surface area ('TPSA': https://www.rdkit.org/docs/GettingStartedInPython.html#descriptor-calculation) of the substrate ('substrate': https://purl.dsmz.de/schema/Substrate) calculated by RDKit software ('RDKit': https://www.rdkit.org/docs/index.html). Unit: square angstroms (Å^2) ('angstrom': http://qudt.org/vocab/unit/ANGSTROM).

  - "substrate_heavy_atom_count"
    - Data type: integer
    - Description: The field 'substrate_heavy_atom_count' indicates the count of heavy atoms ('atom': https://goldbook.iupac.org/terms/view/A00493) in the substrate ('substrate': https://purl.dsmz.de/schema/Substrate).

  - "substrate_hbond_donor_count"
    - Data type: integer
    - Description: The field 'substrate_hbond_donor_count' indicates the count of hydrogen bond donors ('hydrogen bond': https://goldbook.iupac.org/terms/view/H02899) in the substrate ('substrate': https://purl.dsmz.de/schema/Substrate) calculated by RDKit software ('RDKit': https://www.rdkit.org/docs/index.html).

  - "substrate_hbond_acceptor_count"
    - Data type: integer
    - Description: The field 'substrate_hbond_acceptor_count' indicates the count of hydrogen bond acceptors ('hydrogen bond': https://goldbook.iupac.org/terms/view/H02899) in the substrate ('substrate': https://purl.dsmz.de/schema/Substrate) calculated by RDKit software ('RDKit': https://www.rdkit.org/docs/index.html).

  - "substrate_rotatable_bond_count"
    - Data type: integer
    - Description: The field 'substrate_rotatable_bond_count' indicates the count of rotatable bonds ('bond': https://goldbook.iupac.org/terms/view/B00701) in the substrate ('substrate': https://purl.dsmz.de/schema/Substrate) calculated by RDKit software ('RDKit': https://www.rdkit.org/docs/index.html).

  - "substrate_molar_refractivity"
    - Data type: number
    - Description: The field 'substrate_molar_refractivity' indicates the molar refractivity ('molar refractivity': https://old.iupac.org/reports/1997/6905vandewaterbeemd/glossary.html) of the substrate ('substrate': https://purl.dsmz.de/schema/Substrate) calculated by RDKit software ('RDKit': https://www.rdkit.org/docs/index.html). Unit: cubic centimeters per mole (cm^3/mol) ('cubic centimeter': http://qudt.org/vocab/unit/CentiM3; 'mole': http://qudt.org/vocab/unit/MOL).

  - "substrate_possible_structures"
    - Data type: array
    - Description: The field 'substrate_possible_structures' indicates the possible molecular structures ('molecular structure': http://edamontology.org/data_0883) generated for the substrate ('substrate': https://purl.dsmz.de/schema/Substrate).

    Each item in "substrate_possible_structures" is an object containing:

    - "substrate_structure_name"
      - Data type: string
      - Description: The field 'substrate_structure_name' indicates the name of a possible molecular structure ('molecular structure': http://edamontology.org/data_0883) generated for the substrate ('substrate': https://purl.dsmz.de/schema/Substrate).

    - "substrate_structure_energy"
      - Data type: number
      - Description: The field 'substrate_structure_energy' indicates the energy ('energy': http://purl.obolibrary.org/obo/PATO_0001021) of a possible molecular structure ('molecular structure': http://edamontology.org/data_0883) generated for the substrate ('substrate': https://purl.dsmz.de/schema/Substrate). Unit: kilocalories per mole (kcal/mol) ('kilocalorie': http://qudt.org/vocab/unit/KiloCAL; 'mole': http://qudt.org/vocab/unit/MOL).

    - "substrate_structure_max_3d_diameter"
      - Data type: number
      - Description: The field 'substrate_structure_max_3d_diameter' indicates the maximum three-dimensional diameter ('diameter': http://purl.obolibrary.org/obo/PATO_0001334) of a possible molecular structure ('molecular structure': http://edamontology.org/data_0883) generated for the substrate ('substrate': https://purl.dsmz.de/schema/Substrate). Unit: angstroms (Å) ('angstrom': http://qudt.org/vocab/unit/ANGSTROM).

    - "substrate_structure_mean_pairwise_atom_distance"
      - Data type: number
      - Description: The field 'substrate_structure_mean_pairwise_atom_distance' indicates the mean pairwise atom distance ('distance': http://purl.obolibrary.org/obo/PATO_0000040) of a possible molecular structure ('molecular structure': http://edamontology.org/data_0883) generated for the substrate ('substrate': https://purl.dsmz.de/schema/Substrate). Unit: angstroms (Å) ('angstrom': http://qudt.org/vocab/unit/ANGSTROM).

    - "substrate_structure_std_pairwise_atom_distance"
      - Data type: number
      - Description: The field 'substrate_structure_std_pairwise_atom_distance' indicates the standard deviation ('standard deviation': http://purl.obolibrary.org/obo/STATO_0000237) of pairwise atom distances ('distance': http://purl.obolibrary.org/obo/PATO_0000040) of a possible molecular structure ('molecular structure': http://edamontology.org/data_0883) generated for the substrate ('substrate': https://purl.dsmz.de/schema/Substrate). Unit: angstroms (Å) ('angstrom': http://qudt.org/vocab/unit/ANGSTROM).

    - "substrate_structure_asphericity"
      - Data type: number
      - Description: The field 'substrate_structure_asphericity' indicates the asphericity ('asphericity': https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html) of a possible molecular structure ('molecular structure': http://edamontology.org/data_0883) generated for the substrate ('substrate': https://purl.dsmz.de/schema/Substrate) calculated by RDKit software ('RDKit': https://www.rdkit.org/docs/index.html). Unit: dimensionless ('dimensionless': http://qudt.org/vocab/unit/UNITLESS).

    - "substrate_structure_spherocity"
      - Data type: number
      - Description: The field 'substrate_structure_spherocity' indicates the spherocity index ('spherocity index': https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html) of a possible molecular structure ('molecular structure': http://edamontology.org/data_0883) generated for the substrate ('substrate': https://purl.dsmz.de/schema/Substrate) calculated by RDKit software ('RDKit': https://www.rdkit.org/docs/index.html). Unit: dimensionless ('dimensionless': http://qudt.org/vocab/unit/UNITLESS).

    - "substrate_structure_principal_moment_ratio"
      - Data type: number
      - Description: The field 'substrate_structure_principal_moment_ratio' indicates the ratio of the largest to the smallest principal moments of inertia ('moment of inertia': https://goldbook.iupac.org/terms/view/M03954) of a possible molecular structure ('molecular structure': http://edamontology.org/data_0883) generated for the substrate ('substrate': https://purl.dsmz.de/schema/Substrate). Unit: dimensionless ('dimensionless': http://qudt.org/vocab/unit/UNITLESS).

    - "substrate_structure_radius_of_gyration"
      - Data type: number
      - Description: The field 'substrate_structure_radius_of_gyration' indicates the radius of gyration ('radius of gyration': https://goldbook.iupac.org/terms/view/R05121) of a possible molecular structure ('molecular structure': http://edamontology.org/data_0883) generated for the substrate ('substrate': https://purl.dsmz.de/schema/Substrate) calculated by RDKit software ('RDKit': https://www.rdkit.org/docs/index.html). Unit: angstroms (Å) ('angstrom': http://qudt.org/vocab/unit/ANGSTROM).


# Process:

This command processes the input substrates as follows:

1. Parse input substrates
   - Read the input substrate_names string.
   - Split multiple substrates by ';'.
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
   - Compute Morgan fingerprint encoding.
   - Compute selected 2D molecular descriptors, including:
       - substrate atom count
       - substrate molecular weight
       - substrate logP
       - substrate topological polar surface area
       - substrate heavy atom count
       - substrate hydrogen bond donor count
       - substrate hydrogen bond acceptor count
       - substrate rotatable bond count
       - substrate molar refractivity

5. Add hydrogens
   - Automatically add explicit hydrogen atoms to each valid 2D molecule
     before 3D conformer generation using RDKit.

6. Generate possible molecular structures
   - Generate multiple candidate 3D conformers using RDKit embedding.
   - Apply RMS-based pruning to remove highly redundant conformers.

7. Minimize energy
   - Minimize each generated 3D conformer's energy using the UFF force field.
   - Compute conformer energy values for ranking.
   - Compute selected 3D molecular structure descriptors for each generated
     conformation.

8. Rank and organize conformers
   - Sort valid conformers by increasing energy.
   - Rename them in a consistent order such as glucose_1, glucose_2, etc.

9. Save outputs
   - Save each valid 3D substrate conformation as an individual SDF file.
   - Generate and save a JSON report summarizing resolved substrate information,
     computed features, and generated structure metadata.


# common errors and solutions:

- "Invalid substrate generation parameters. Require: max_synonyms (1–200), fp_radius (1–5), n_bits (1–2048), num_confs (1–20), prune_rms (0–5]."
  - Cause: One or more generation parameters are outside the supported range.
  - Solution: Use values within the documented ranges, for example `--max_synonyms 20 --fp_radius 2 --n_bits 512 --num_confs 5 --prune_rms 0.5`.

- "substrate_names is empty after parsing."
  - Cause: The `-s` value is empty or contains only separators and whitespace.
  - Solution: Provide at least one substrate name or SMILES string, such as `-s "glucose"` or `-s "CCO"`.

- "Duplicate substrate name detected"
  - Cause: Two input entries produce the same internal substrate name. This can happen when substrate names are repeated, or when a name such as `smiles1` conflicts with an automatically assigned SMILES name.
  - Solution: Remove duplicate substrate names, avoid names such as `smiles1` for named substrates, or run repeated substrates separately if independent output folders are needed.

- "Failed to obtain SMILES for substrate"
  - Cause: The substrate name could not be resolved to a SMILES string through ChEBI or PubChem.
  - Solution: Check the spelling, use a more specific substrate name, increase `--max_synonyms`, or provide the SMILES string directly with `-s`.

- "Invalid SMILES"
  - Cause: A direct SMILES input cannot be parsed by RDKit.
  - Solution: Check the SMILES syntax and use a valid canonical or isomeric SMILES string.

- "Failed to generate Mol(2D) for substrate"
  - Cause: RDKit could not convert the resolved SMILES into a valid 2D molecular object.
  - Solution: Check whether the SMILES string represents a supported small molecule and try a corrected SMILES input.

- "Failed to generate fingerprint for substrate"
  - Cause: Morgan fingerprint generation failed for the RDKit molecule.
  - Solution: Check the substrate SMILES and use standard fingerprint parameters, such as `--fp_radius 2 --n_bits 512`.

- "Failed to generate 2D descriptors for substrate"
  - Cause: RDKit could not calculate the required 2D molecular descriptors.
  - Solution: Check that the substrate can be parsed as a chemically valid small molecule and rerun with a corrected name or SMILES string.

- "Failed to save SDF file."
  - Cause: RDKit attempted to write a generated 3D conformation, but the SDF file was not created or was empty.
  - Solution: Check that the output directory is writable and that there is enough disk space.

- "Failed to save Mol(3D) to SDF file"
  - Cause: Writing a generated 3D substrate conformation to SDF failed because of a filesystem problem or an invalid generated molecule.
  - Solution: Check the output directory permissions and review earlier 3D generation messages in `log.txt`.

- "Filename too long"
  - Cause: The generated report or SDF file name is longer than the supported filename limit.
  - Solution: Use shorter substrate names, fewer substrates in one run, or direct SMILES inputs that use shorter automatic names.

- "Failed to write report JSON to"
  - Cause: The JSON report could not be written to the output directory because of a filesystem, permission, or disk-space problem.
  - Solution: Check that the `-o` output directory path is writable and that there is enough disk space.

- Substrate feature generation failed
  - Cause: SMILES resolution, RDKit molecule construction, fingerprint generation, or 2D descriptor calculation failed before a schema-valid report could be built.
  - Solution: Review the specific error above this summary in `log.txt`, then correct the substrate name, provide a direct SMILES string, or use standard feature parameters.

- No SDF files are generated
  - Cause: 3D conformer generation, minimization, energy calculation, or 3D descriptor calculation failed for all candidate conformers. The JSON report can still be valid if substrate identity, fingerprint, and 2D descriptors were generated successfully.
  - Solution: Check `log.txt` for 3D generation warnings, use a smaller `--prune_rms` to prune less aggressively, increase `--num_confs`, or provide a simpler valid SMILES string.

- Output files are missing
  - Cause: The command failed before all outputs were written, or the output directory is not the one expected.
  - Solution: Check `log.txt`, confirm the `-o` output directory, and rerun after fixing earlier errors.

- Output file names are different from expected
  - Cause: Output names use resolved substrate names. For direct SMILES inputs, names such as `smiles1` and `smiles2` are used. For multiple substrates, the JSON report name combines the resolved names in input order.
  - Solution: Check the resolved substrate names in the JSON report and `log.txt`, then look for files named `substrate_report_...json` and SDF files named with the resolved substrate name plus a conformer index.


# dependencies:

- RDKit
- requests
- urllib3
- NumPy


# references:

- RDKit:
  https://www.rdkit.org/

- RDKit Book:
  https://www.rdkit.org/docs/RDKit_Book.html

- PubChem PUG REST API:
  https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest

- ChEBI:
  https://www.ebi.ac.uk/chebi/
