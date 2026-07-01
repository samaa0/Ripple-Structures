# Third-Party Notices

This repository is a curated public release of Ripple Structures.

## Firmware

The Arduino firmware includes DW3000 support files that retain upstream
Decawave/Qorvo copyright notices in the source headers. Those files are not
relicensed by the MIT license for project-authored code. Review those files and
the relevant DW3000 SDK/module vendor terms before redistribution or commercial
reuse.

## Python

The Python components depend on open-source packages including NumPy, SciPy,
scikit-learn, FilterPy, VisPy, Requests, Joblib, PyTorch, LightGBM, XGBoost,
Optuna, Matplotlib, Pandas, and Keras. Package code is not vendored in this
repository; install dependencies from the relevant `requirements.txt` files.

## Unity

The Unity project uses packages declared in `unity/ripple-visualizer/Packages`,
including Unity UI, TextMeshPro, Timeline, Visual Scripting, and Unity module
packages. Unity package source is resolved by the Unity Package Manager.

## Hardware License

The PCB and case design files are released under CERN-OHL-S-2.0. The canonical
license text is included at `hardware/LICENSE-CERN-OHL-S-2.0.txt`.
