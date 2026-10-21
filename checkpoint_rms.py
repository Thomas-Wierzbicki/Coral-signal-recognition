"""Sichert den lokalen Stand als gepruefte ZIP und testet danach den RMS-Coral-Prozess."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
from signal_preprocessing import model_preprocessing

ROOT = Path(__file__).resolve().parent


def git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments],
                                   stderr=subprocess.PIPE)


def environment_info():
    import importlib.metadata
    result = dict(python=sys.version, executable=sys.executable,
                  packages={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()
                            if d.metadata.get("Name")})
    training = Path(sys.prefix).parent / "coral-train" / "conda-meta"
    if training.is_dir():
        result["coral_train_conda_packages"] = []
        for path in sorted(training.glob("*.json")):
            package = json.loads(path.read_text(encoding="utf-8"))
            result["coral_train_conda_packages"].append(
                {k:package.get(k) for k in ("name","version","build")})
    return result


def validate_rms_artifacts(root):
    required = [
        "models/signal-rms/signal_float.h5", "models/signal-rms/signal_int8.tflite",
        "models/signal-rms/preprocessing.json", "models/signal-rms/report.json",
        "models/signal-rms/compiled/signal_int8_edgetpu.tflite",
        "models/signal-rms/compiled/preprocessing.json",
        "data/normalized/train.npz", "data/normalized/validation.npz",
        "data/normalized/test.npz", "data/normalized/metadata.json"]
    missing = [p for p in required if not (root/p).is_file()]
    if missing:
        raise ValueError("Fehlende Dateien: " + ", ".join(missing))
    model=root/"models/signal-rms/compiled/signal_int8_edgetpu.tflite"
    if model_preprocessing(model)!="rms-v1":
        raise ValueError("Manifest neben kompiliertem Modell muss rms-v1 angeben")
    metadata=json.loads((root/"data/normalized/metadata.json").read_text(encoding="utf-8"))
    if metadata.get("preprocessing")!="rms-v1":
        raise ValueError("Normalisierte Datenmetadaten fehlen oder passen nicht")


def checkpoint_files(root):
    selected = {root / p.decode("utf-8") for p in git(root,"ls-files","-z").split(b"\0") if p}
    for pattern in ("*.py","*.md","requirements*.txt"):
        selected.update(root.glob(pattern))
    for directory in ("models","data"):
        selected.update(p for p in (root/directory).rglob("*") if p.is_file())
    return sorted(p for p in selected if p.is_file())


def create_checkpoint(root, output):
    root=root.resolve()
    output=output.resolve()
    # Archiv absichtlich ausserhalb des Projekts, damit es nicht mitgeloescht wird.
    if output==root or root in output.parents:
        raise ValueError("Sicherungsordner muss ausserhalb des Projektordners liegen")
    output.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    archive=output/("coral-rms-"+stamp+".zip")
    partial=archive.with_suffix(".zip.partial")
    metadata=dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  git_head=git(root,"rev-parse","HEAD").decode().strip(),
                  git_status=git(root,"status","--short").decode("utf-8",errors="replace"),
                  environment=environment_info())
    checksums={}
    restore=(
        "Sicherung in einen neuen Ordner entpacken. project/ enthaelt den Dateistand, "
        "einschliesslich data/ und models/. Beide RMS-Modelldateien und preprocessing.json "
        "zusammenlassen. repository.bundle enthaelt Git-Commits und Referenzen.\n"
        "Optional: git clone repository.bundle recovered; danach den Inhalt von project/ "
        "in recovered kopieren, um den gesicherten Arbeitsstand wiederherzustellen.\n"
        "Python/Conda, Windows-Edge-TPU-Laufzeit und der Ubuntu-Compiler sind nicht "
        "als installierte Programme enthalten. environment.json und README beschreiben "
        "die Umgebung. SHA256SUMS.json beschreibt die Inhalte des Archivs.\n")
    with tempfile.TemporaryDirectory() as temporary:
        bundle=Path(temporary)/"repository.bundle"
        git(root,"bundle","create",str(bundle),"--all")
        with zipfile.ZipFile(partial,"x",compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6,allowZip64=True) as z:
            for path in checkpoint_files(root):
                name="project/"+path.relative_to(root).as_posix()
                digest=hashlib.sha256()
                with path.open("rb") as source,z.open(name,"w",force_zip64=True) as target:
                    while True:
                        block=source.read(1024*1024)
                        if not block:
                            break
                        digest.update(block)
                        target.write(block)
                checksums[name]=digest.hexdigest()
            z.write(bundle,"repository.bundle")
            checksums["repository.bundle"]=hashlib.sha256(bundle.read_bytes()).hexdigest()
            extras={"environment.json":json.dumps(metadata,indent=2),
                    "RESTORE.txt":restore,
                    "working-tree.patch":git(root,"diff","HEAD","--binary").decode("utf-8",errors="replace")}
            for name,content in extras.items():
                raw=content.encode("utf-8")
                z.writestr(name,raw)
                checksums[name]=hashlib.sha256(raw).hexdigest()
            z.writestr("SHA256SUMS.json",json.dumps(checksums,indent=2))
        # Alle archivierten Dateien zuruecklesen und gegen die gespeicherten Hashes pruefen.
        with zipfile.ZipFile(partial) as z:
            for name,expected in checksums.items():
                digest=hashlib.sha256()
                with z.open(name) as source:
                    for block in iter(lambda:source.read(1024*1024),b""):
                        digest.update(block)
                if digest.hexdigest()!=expected:
                    raise ValueError("Archivpruefung fehlgeschlagen: "+name)
    partial.rename(archive)
    return archive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT.parent/"Coral-signal-recognition-backups")
    parser.add_argument("--backup-only",action="store_true")
    args=parser.parse_args()
    try:
        validate_rms_artifacts(ROOT)
        print("Sichere Quellstand, beide Modellordner, Daten und JSON-Dateien ...",flush=True)
        archive=create_checkpoint(ROOT,args.output)
        print("Sicherung inklusive SHA-256-Pruefung erfolgreich:\n"+str(archive),flush=True)
        print("Das ist eine lokale Sicherung, kein Upload auf GitHub.",flush=True)
        if args.backup_only:
            return 0
        command=[sys.executable,"-u",str(ROOT/"live_demo.py"),"--model",
                 str(ROOT/"models/signal-rms/compiled/signal_int8_edgetpu.tflite"),
                 "--preprocessing","rms-v1"]
        log=archive.with_suffix(".process-test.txt")
        print("\nStarte RMS-Prozesstest; Protokoll: "+str(log),flush=True)
        with log.open("x",encoding="utf-8") as stream:
            stream.write("Command: "+json.dumps(command)+"\n")
            stream.flush()
            with subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True,encoding="utf-8",
                                  errors="replace") as process:
                try:
                    for line in process.stdout:
                        print(line,end="",flush=True)
                        stream.write(line)
                        stream.flush()
                    code=process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            stream.write("\nExit code: "+str(code)+"\n")
        print("Sicherung bleibt erhalten. Testprotokoll liegt daneben.",flush=True)
        return code
    except KeyboardInterrupt:
        print("\nAbgebrochen. Bereits fertiggestellte Sicherung bleibt erhalten.",file=sys.stderr)
        return 130
    except (OSError,ValueError,subprocess.CalledProcessError) as error:
        print("Sicherung/Test fehlgeschlagen: "+str(error),file=sys.stderr)
        return 1


if __name__=="__main__":
    sys.exit(main())
