"""Unpack the compressed tensor store into the directory layout the notebooks expect.

The repository ships instances as compressed ``.npz`` (float32) under
``data/tensors/<family>/<N>_<index>.npz``. The research notebooks, kept close to
their original form, load ``.npy`` files from per-family folders with the
original file names. Running this script once recreates that layout under
``data/legacy``, after which the notebooks run unchanged with
``PUBO_DATA=<repo>/data/legacy``.

Expect roughly 5 GB of uncompressed output.
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "tensors")
DST = os.path.join(ROOT, "data", "legacy")

# family -> (legacy folder, file-name template keyed by the "<N>_<index>" stem)
LAYOUT = {
    "dense":         ("Dense",         "K_dense_{stem}_.npy"),
    "sparse_random": ("Sparse random", "K_sparse_random_{stem}_.npy"),
    "sk_symmetric":  ("sk_symmetric",  "{stem}.npy"),
    "n200":          ("200/J_T_h",     "T_{index}.npy"),
}


def main() -> None:
    for family, (folder, template) in LAYOUT.items():
        src = os.path.join(SRC, family)
        if not os.path.isdir(src):
            print(f"skipping {family}: not present")
            continue
        out = os.path.join(DST, folder)
        os.makedirs(out, exist_ok=True)
        n = 0
        for fn in sorted(os.listdir(src)):
            if not fn.endswith(".npz"):
                continue
            stem = fn[:-4]
            if family == "n200":
                # stems are "n200_<index>"; the notebooks index instances only
                index = stem.rsplit("_", 1)[-1]
                name = template.format(index=index, stem=stem)
            else:
                name = template.format(stem=stem, index=stem)
            with np.load(os.path.join(src, fn)) as f:
                np.save(os.path.join(out, name), f["T"])
            n += 1
        print(f"{family:<14} -> data/legacy/{folder}  ({n} files)")
    print("\nNow run the notebooks with PUBO_DATA=" + DST)


if __name__ == "__main__":
    main()
