"""RadioML-Projektstand lokal als ZIP mit SHA256-Pruefung sichern; kein Git erforderlich."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parent


def required_artifacts(root):
    model=root/'models/radioml2018-iq-v1'
    required=[model/name for name in ('radioml_float.h5','radioml_int8.tflite',
        'preprocessing.json','labels.txt','report.json','test_predictions.npz',
        'compiled/radioml_int8_edgetpu.tflite','compiled/coral_report.json',
        'compiled/coral_predictions.npz')]
    for variant in ('radioml2018-v1','radioml2018-iq-v1'):
        required.extend(root/'data'/variant/name for name in
                        ('train.npz','validation.npz','test.npz','metadata.json','IMPORT_COMPLETE'))
    missing=[str(p.relative_to(root)) for p in required if not p.is_file()]
    if missing:
        raise ValueError('Fehlende Dateien: '+', '.join(missing))
    manifest=json.loads((model/'preprocessing.json').read_text(encoding='utf-8'))
    if (manifest.get('representation')!='iq-rms-v1' or manifest.get('input_shape')!=[1,1024,2]
            or manifest.get('target_rms')!=0.5):
        raise ValueError('I/Q-Modellmanifest passt nicht')


def selected_files(root):
    # Projektdateien im Stamm und alle Daten/Modelle; keine Git-Installation noetig.
    paths=[p for p in root.iterdir() if p.is_file() and
           (p.suffix.lower() in ('.py','.md','.txt','.json','.toml','.yml','.yaml')
            or p.name in ('.gitignore','.gitattributes'))]
    for name in ('data','models'):
        paths.extend(p for p in (root/name).rglob('*') if p.is_file())
    for path in sorted(set(paths)):
        if path.is_symlink() or root not in path.resolve().parents:
            raise ValueError('Verknuepfung ausserhalb/innerhalb des Projekts nicht archiviert: '+str(path))
        yield path


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        hashes=json.loads(archive.read('SHA256SUMS.json'))
        if set(archive.namelist())!=set(hashes)|{'SHA256SUMS.json'}:
            raise ValueError('Archivinhalte stimmen nicht mit Hashliste ueberein')
        for name,expected in hashes.items():
            digest=hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    digest.update(block)
            if digest.hexdigest()!=expected:
                raise ValueError('SHA256-Pruefung fehlgeschlagen: '+name)
    return len(hashes)


def create_checkpoint(root,output,source=None):
    root=root.resolve()
    output=output.resolve()
    if output==root or root in output.parents:
        raise ValueError('Ziel muss ausserhalb des Projektordners liegen')
    output.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    archive=output/('coral-radioml-'+stamp+'.zip')
    partial=archive.with_suffix('.zip.partial')
    files=[(p,'project/'+p.relative_to(root).as_posix()) for p in selected_files(root)]
    if source is not None:
        source=source.resolve()
        if not source.is_file():
            raise ValueError('HDF5-Quelldatei fehlt')
        files.append((source,'external-source/'+source.name))
    hashes={}
    environment=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version,executable=sys.executable,
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()
                  if d.metadata.get('Name')},source_hdf5_included=source is not None,
        scope='root code/docs/config, recursive data/models; no Git history or installed software')
    restore='''In einen NEUEN Ordner entpacken; project/ enthaelt Quellcode, Daten und Modelle.
SHA256SUMS.json prueft alle Nutzdateien im Archiv. ZIP vor Wiederherstellung pruefen:
python checkpoint_radioml.py --verify "PFAD_ZUR_ZIP"
Anaconda-Umgebungen, Treiber und Ubuntu-Compiler separat installieren (README).
Git-Historie ist nicht enthalten. environment.json dokumentiert das aktive Python.
HDF5-Original nur enthalten, wenn external-source/ existiert; sonst separat aufbewahren.
Modellordner samt preprocessing.json, labels.txt und Berichten zusammenlassen.
Unter Windows, Anaconda Prompt, Projektordner: conda activate coral
python evaluate_radioml_iq_coral.py
'''
    with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for path,name in files:
            before=path.stat()
            digest=hashlib.sha256()
            with path.open('rb') as inp,z.open(name,'w',force_zip64=True) as out:
                for block in iter(lambda:inp.read(1024*1024),b''):
                    digest.update(block)
                    out.write(block)
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise ValueError('Datei waehrend Sicherung geaendert: '+str(path))
            hashes[name]=digest.hexdigest()
        for name,content in {'environment.json':json.dumps(environment,indent=2),'RESTORE.txt':restore}.items():
            raw=content.encode('utf-8')
            z.writestr(name,raw)
            hashes[name]=hashlib.sha256(raw).hexdigest()
        z.writestr('SHA256SUMS.json',json.dumps(hashes,indent=2))
    verify_archive(partial)
    partial.rename(archive)
    return archive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT.parent/'Coral-signal-recognition-backups')
    parser.add_argument('--include-source',type=Path,help='Optional: grosse HDF5-Datei ebenfalls archivieren')
    parser.add_argument('--verify',type=Path,help='Nur vorhandene ZIP pruefen')
    args=parser.parse_args()
    try:
        if args.verify:
            print('ZIP geprueft:',verify_archive(args.verify),'Dateien')
            return 0
        required_artifacts(ROOT)
        print('Sichere Modelle, Daten, JSON, Code und Dokumentation. Bitte keine Dateien veraendern.',flush=True)
        archive=create_checkpoint(ROOT,args.output,args.include_source)
        print('Sicherung erfolgreich, alle archivierten Nutzdateien mit SHA256 geprueft:\n'+str(archive))
        print('Lokale Sicherung; kein Upload. HDF5-Original enthalten:',args.include_source is not None)
        return 0
    except (OSError,ValueError,KeyError,zipfile.BadZipFile) as error:
        print('Sicherung/Pruefung fehlgeschlagen: '+str(error),file=sys.stderr)
        return 1


if __name__=='__main__':
    sys.exit(main())
