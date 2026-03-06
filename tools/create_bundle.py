import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

PAGES_DIR = ROOT_DIR / "pages"
INGESTION_DIR = ROOT_DIR / "ingestion_program"
SCORING_DIR = ROOT_DIR / "scoring_program"
DEV_PHASE_DIR = ROOT_DIR / "dev_phase"
SOLUTION_DIR = ROOT_DIR / "solution"

BUNDLE_FILES = [
    ROOT_DIR / "competition.yaml",
    ROOT_DIR / "logo.png",
]


if __name__ == "__main__":
    with zipfile.ZipFile(ROOT_DIR / "bundle.zip", mode="w") as bundle:
        for f in BUNDLE_FILES:
            rel = f.relative_to(ROOT_DIR)
            print(rel)
            bundle.write(f, rel)
        for dirpath in [
            INGESTION_DIR, SCORING_DIR, PAGES_DIR,
            DEV_PHASE_DIR, SOLUTION_DIR
        ]:
            assert dirpath.exists(), (
                f"{dirpath} does not exist. Make sure you followed all "
                "the instructions in the README before creating the bundle."
            )
            for f in dirpath.rglob("*"):
                if not f.is_file():
                    continue
                if f.name.startswith(".") or f.name.endswith(".pyc"):
                    continue
                rel = f.relative_to(ROOT_DIR)
                print(rel)
                bundle.write(f, rel)

    size = (ROOT_DIR / "bundle.zip").stat().st_size / 1024 / 1024
    print(f"\nbundle.zip created: {size:.1f} MB")
