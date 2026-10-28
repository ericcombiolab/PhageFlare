# PhageFlare

PhageFlare is a deep-learning framework for multilabel functional annotation of bacteriophage proteins. It uses a frozen [ESM-C 600M](https://github.com/evolutionaryscale/esm) protein language model, function-conditioned residue pooling, and optional genomic context to predict 45 functional labels. PhageFlare is designed to complement alignment- and structure-based annotation, including for proteins without significant structural matches to annotated references.

This repository provides a trained checkpoint and an inference script that accepts either FASTA or JSON Lines (JSONL) input.

## Requirements

- Python 3.11
- PyTorch and the dependencies in `environment.yml`
- A CUDA-capable GPU is strongly recommended; inference falls back to CPU when CUDA is unavailable, but will be substantially slower
- Internet access on the first run to obtain the pretrained ESM-C weights, unless they are already cached

The supplied environment targets a Linux/CUDA setup. Create and activate it from the repository root:

```bash
conda env create -f environment.yml
conda activate phageflare
```

## Quick start

Run inference from the repository root so that the default checkpoint and threshold paths resolve correctly:

```bash
python inference.py \
  --target_data_path sample_data/test.jsonl \
  --output_path outputs/predictions.csv
```

This uses:

- checkpoint: `checkpoint/PhageFlare.pt`
- thresholds: `binarization_threshold_for_each_label_high_precision.json`
- batch size: `1`

For a collection of protein sequences without genomic context, FASTA input is sufficient:

```bash
python inference.py \
  --target_data_path proteins.fasta \
  --output_path outputs/predictions.csv \
  --batch_size 4
```

Increase `--batch_size` to improve throughput if memory permits. Reduce it if you encounter out-of-memory errors; long proteins and genomic-context proteins also increase memory use.

## Input data

PhageFlare chooses the input parser from the file contents, not the filename extension. If the first non-empty line starts with `>`, the file is parsed as FASTA; otherwise, it is parsed as JSONL.

For both formats:

- Each record represents one target protein.
- `seq_id` identifies the target and is copied to the output.
- `prot_seq` is the target amino-acid sequence and must be non-empty.
- Target sequences longer than **2,048 amino acids** are rejected.
- Use amino-acid sequences without stop-codon markers (`*`). For consistency with the training-data preprocessing, avoid noncanonical symbols `X`, `B`, `Z`, and `J`.
- Input order is preserved in the output.

### Option 1: FASTA (target proteins only)

FASTA is the simplest input format when genomic context is unavailable. PhageFlare supplies empty context and uses the target-sequence predictor without a context correction.

```fasta
>phageA_00001 hypothetical protein
MKTLLKTLAAGLAAAGLTAPHAFAKDGKVTW
>phageA_00002
MNNLFKKIVGIIGPTGVGKTDIAAYIAER
```

FASTA rules:

- The sequence identifier is the first whitespace-delimited token after `>`. In the example above, it is `phageA_00001`; the description is not used.
- Sequences may span multiple lines. Blank lines are ignored and whitespace within sequence lines is removed.
- Every record must have a non-empty identifier and sequence.
- Use a unique identifier for each record to make the output easier to join back to the input.

### Option 2: JSONL (with optional genomic context)

JSONL contains one JSON object per line; the entire file is **not** a JSON array. Each object requires `seq_id` and `prot_seq`:

```jsonl
{"seq_id":"phageA_00001","prot_seq":"MKTLLKTLAAGLAAAGLTAPHAFAKDGKVTW"}
{"seq_id":"phageA_00002","prot_seq":"MNNLFKKIVGIIGPTGVGKTDIAAYIAER"}
```

To use genomic context, add up to three upstream and three downstream protein sequences for each target. The index `0` denotes the immediate neighbor, `1` the next neighbor, and `2` the third neighbor:

| Field | Type | Meaning |
|---|---|---|
| `seq_id` | string | Target-protein identifier; required |
| `prot_seq` | string | Target amino-acid sequence; required |
| `left_seq_0`, `left_seq_1`, `left_seq_2` | string | Upstream/left protein sequences, from nearest to farthest; optional |
| `right_seq_0`, `right_seq_1`, `right_seq_2` | string | Downstream/right protein sequences, from nearest to farthest; optional |

Here, left/right refer to the gene order used to construct the dataset. Preserve the same ordering convention for all records.

Example with all six context slots populated:

```jsonl
{"seq_id":"phageA_00002","prot_seq":"MNNLFKKIVGIIGPTGVGKTDIAAYIAER","left_seq_0":"MKTLLKTLAAGLAAAGLTAPHAFAKDGKVTW","left_seq_1":"MSDKQIEALVKQLEAGDA","left_seq_2":"MKKAFALATAGLFGAGA","right_seq_0":"MTKFDKLAELKPELVAD","right_seq_1":"MGKQVTLKDAFVDGKTA","right_seq_2":"MSVAKDLAQALDEAKLA"}
```

Missing context slots may be omitted, set to `""`, or set to `null`. For example, this record has only its two immediate neighbors:

```jsonl
{"seq_id":"phageB_00007","prot_seq":"MNNLFKKIVGIIGPTGVGKTDIAAYIAER","left_seq_0":"MKTLLKTLAAGLAAAGLTAPHAFAKDGKVTW","right_seq_0":"MTKFDKLAELKPELVAD"}
```

Context columns are optional for each record. If no context columns are present anywhere in the JSONL file, all proteins are treated as context-free. Additional columns, including `consistent_label`, `inconsistent_label`, `domain`, neighbor IDs, and neighbor labels, are ignored during inference; ground-truth annotations are not required. See `sample_data/test.jsonl` for a full example in the training-data layout.

## Output format

The output is a CSV file in the same row order as the input. Parent directories are created automatically.

| Column | Description |
|---|---|
| `index` | Zero-based input-record index |
| `seq_id` | Target identifier copied from the input |
| `predicted_labels` | Predicted positive labels, separated by semicolons (`;`) |
| `<label>_probability` | Sigmoid probability for each of the 45 functional labels |

An empty `predicted_labels` value means that no label passed the decision rules; it can be interpreted as `unknown`.

Probabilities are the model's raw per-label probabilities. They are not changed by the subsequent hierarchy filter, so a probability column can remain high even when that label is removed from `predicted_labels` because its required parent label was negative.

<details>
<summary>Supported functional labels</summary>

```text
DNA-associated, DNA_polymerase, RNA-associated, adsorption-related,
annealing, anti-restriction, baseplate, capsid, cell_wall_depolymerase,
collar, crispr, defense_systems, ejection, endolysin, head-tail_joining,
helicase, holin, integration, lysis, lysis_inhibitor, major_capsid,
major_tail, minor_capsid, minor_tail, nuclease, nucleotide_metabolism,
packaging_assembly, phosphorylation, portal, primase, pvp, reductase,
replication_initiation, sir2, spanin, super_infection, tail,
tail_appendage, tail_sheath, terminase, toxin, transcriptional_activator,
transcriptional_regulator, transcriptional_repressor, transferase
```

</details>

## Decision thresholds and label hierarchy

PhageFlare performs multilabel prediction: one protein can receive zero, one, or multiple labels. Each label has its own probability threshold. Two threshold files are provided:

- `binarization_threshold_for_each_label_high_precision.json` — the default, precision-oriented threshold set.
- `binarization_threshold_for_each_label.json` — validation-selected, F1-oriented thresholds.

To use the F1-oriented thresholds:

```bash
python inference.py \
  --target_data_path proteins.fasta \
  --thresholds_file binarization_threshold_for_each_label.json \
  --output_path outputs/predictions_f1.csv
```

After thresholding, PhageFlare enforces its functional-label hierarchy: a child label is retained only when all of its mapped ancestor labels are also positive. For example, `major_tail` requires `tail`, which in turn requires `pvp`.

## Command-line reference

```text
usage: inference.py [-h] --target_data_path TARGET_DATA_PATH
                    [--checkpoint_path CHECKPOINT_PATH]
                    [--thresholds_file THRESHOLDS_FILE]
                    --output_path OUTPUT_PATH [--batch_size BATCH_SIZE]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--target_data_path` | Yes | — | Input FASTA or JSONL file |
| `--output_path` | Yes | — | Destination CSV file |
| `--checkpoint_path` | No | `checkpoint/PhageFlare.pt` | Trained PhageFlare checkpoint |
| `--thresholds_file` | No | `binarization_threshold_for_each_label_high_precision.json` | JSON object mapping every label to a numeric threshold |
| `--batch_size` | No | `1` | Number of target proteins per inference batch |

The script automatically selects CUDA when available and otherwise uses CPU.

## Troubleshooting

- **`Long sequences found in the dataset`**: remove or split inputs appropriately so that every target sequence is at most 2,048 residues. PhageFlare does not automatically truncate sequences.
- **CUDA out of memory**: lower `--batch_size` to `1`, shorten the batch's sequence lengths where biologically appropriate, or use a GPU with more memory.
- **Missing labels in a custom thresholds file**: the JSON file must contain every label stored in the checkpoint. Extra labels are ignored.
- **Missing checkpoint or thresholds**: run from the repository root, or pass absolute paths with `--checkpoint_path` and `--thresholds_file`.
- **Unexpected JSON parsing errors**: ensure each JSONL line is a complete JSON object and that the file does not contain a surrounding `[...]` array or trailing commas.

## Reference

Chonghao Wang, Yu Xu, Zeming Li, Louxin Zhang, and Lu Zhang. *Illuminating phage dark matter beyond the reach of structural alignments with PhageFlare*.
