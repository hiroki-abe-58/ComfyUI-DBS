"""Model location: `<ComfyUI models>/dbs/<model name>/` with config.json and model.safetensors (plus any path added for `dbs` in
extra_model_paths.yaml), i.e. the files of the Hugging Face repositories JusperLee/DBS-TIGER-DnR and JusperLee/DBS-BSRNN-DnR.
Nothing is downloaded at run time.
"""

from __future__ import annotations

import os

FOLDER = "dbs"


def register() -> None:
    import folder_paths

    folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER), is_default=False)


def model_dir(name: str, files: tuple[str, ...]) -> str:
    """The `<name>` folder of the first registered dbs folder that holds all of `files`. `name` must be a known model name (the
    caller checks it against the fixed list), so it cannot leave the models folder."""
    import folder_paths

    for base in folder_paths.get_folder_paths(FOLDER):
        folder = os.path.join(base, name)
        if all(os.path.isfile(os.path.join(folder, f)) for f in files):
            return folder
    raise FileNotFoundError(f"models/{FOLDER}/{name}/ does not hold {', '.join(files)} (see the README for the download command)")
