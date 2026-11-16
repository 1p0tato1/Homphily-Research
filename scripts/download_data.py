"""Télécharge les datasets dans data/ (ignoré par git).

    python scripts/download_data.py                  # Cornell, Texas, Wisconsin, Actor (~110 Mo)
    python scripts/download_data.py amazon           # Amazon (255 Mo)
    python scripts/download_data.py twitter          # TEG-DB twitter
"""
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

DATA = Path(__file__).resolve().parents[1] / "data"
HETGB = {"cornell": "Cornell", "texas": "Texas", "wisconsin": "Wisconsin", "actor": "Actor", "amazon": "Amazon"}
DEFAULT = ["cornell", "texas", "wisconsin", "actor"]


def get_hetgb(key):
    target = DATA / "hetgb" / f"{HETGB[key]}.npz"
    if target.exists():
        return print(f"déjà là : {target.name}")
    files = [f for f in list_repo_files("jinxi0219/HeTGB", repo_type="dataset") if f.lower().endswith(".npz")]
    found = [f for f in files if key in f.lower()]
    if not found:
        sys.exit(f"{key} introuvable, fichiers vus : {files}")
    print(f"téléchargement : {found[0]}")
    got = Path(hf_hub_download("jinxi0219/HeTGB", found[0], repo_type="dataset", local_dir=DATA / "hetgb"))
    if got != target:
        shutil.move(str(got), target)


def get_twitter():
    print("téléchargement : TEG-DB twitter")
    snapshot_download("ZhuofengLi/TEG-Datasets", repo_type="dataset", allow_patterns=["twitter/*"], local_dir=DATA / "teg_db")


for name in sys.argv[1:] or DEFAULT:
    get_twitter() if name == "twitter" else get_hetgb(name)
