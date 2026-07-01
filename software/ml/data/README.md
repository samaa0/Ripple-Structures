# Data Files

Raw training CSVs are intentionally not included in this public release.

The ML scripts expect sequence-level UWB data with columns such as:

- `sequence`
- `timestamp`
- `x`
- `y`
- `z`
- `label`

Feature-branch CSVs should include `sequence`, `label`, and extracted numeric
features. Place private/local datasets here when reproducing the experiments;
the repository `.gitignore` keeps CSV data files out of public commits by
default.
