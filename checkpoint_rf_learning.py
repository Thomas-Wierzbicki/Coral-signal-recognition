"""Back up the WSL rf-learning workspace, including models/data, with SHA256 verification."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

EXCLUDED={'.venv','venv','env','.git','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache'}


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def signature(path):
    info=path.stat()
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)


def collect(root):
    files=[]
    for directory,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=sorted(name for name in dirs if name not in EXCLUDED)
        for name in dirs:
            if (Path(directory)/name).is_symlink():raise ValueError('Symlink directory: '+str(Path(directory)/name))
        for name in sorted(names):
            path=Path(directory)/name
            if path.is_symlink() or not path.is_file():raise ValueError('Not a regular file: '+str(path))
            files.append(path)
    return files


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        hashes=json.loads(archive.read('SHA256SUMS.json'))
        if len(names)!=len(set(names)) or set(names)!=set(hashes)|{'SHA256SUMS.json'}:
            raise ValueError('Archive entries differ from manifest')
        for name,expected in hashes.items():
            value=hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
            if value.hexdigest()!=expected:raise ValueError('Archive SHA256 mismatch: '+name)
    return len(hashes)


def create_backup(root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    if not all((root/name).is_dir() for name in ['project','data','models']):
        raise ValueError('Expected rf-learning with project/, data/ and models/')
    if output==root or root in output.parents:raise ValueError('Backup destination must be outside rf-learning')
    files=collect(root);before={path:signature(path) for path in files}
    output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output).free<sum(v[2] for v in before.values())*1.05+64*1024*1024:
        raise ValueError('Not enough free destination space for a verified backup')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ')
    archive=output/('rf-learning-'+stamp+'.zip');partial=archive.with_suffix('.zip.partial')
    try:
        revision=subprocess.run(['git','-C',str(root/'project'),'rev-parse','HEAD'],capture_output=True,text=True,timeout=10)
        revision=revision.stdout.strip() if revision.returncode==0 else None
    except (OSError,subprocess.SubprocessError):revision=None
    environment=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_root=str(root),
        python=sys.version,executable=sys.executable,git_head=revision,
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')},
        scope='all regular files under rf-learning, except named environments/git/cache directories',
        excluded_directory_names=sorted(EXCLUDED),
        installed_environment_binaries_included=False,
        external_hdf5_included=False,
        note='Only files actually under source_root included; external Windows data and backups are not copied')
    hashes={}
    with zipfile.ZipFile(partial,'x',allowZip64=True) as z:
        for number,path in enumerate(files,1):
            name='rf-learning/'+path.relative_to(root).as_posix()
            info=zipfile.ZipInfo.from_file(path,name)
            info.compress_type=zipfile.ZIP_DEFLATED if path.suffix.lower() in {'.py','.md','.json','.txt','.csv','.jsonl','.log','.yaml','.yml'} else zipfile.ZIP_STORED
            value=hashlib.sha256()
            with path.open('rb') as source,z.open(info,'w',force_zip64=True) as target:
                for block in iter(lambda:source.read(1024*1024),b''):
                    value.update(block);target.write(block)
            if signature(path)!=before[path]:raise ValueError('Source changed while backing up: '+str(path))
            hashes[name]=value.hexdigest()
            if number%50==0:print('Archiviert:',number,'/',len(files),flush=True)
        restore=('Verify ZIP with checkpoint_rf_learning.py --verify PATH.\n'
                 'Extract into a NEW empty directory. rf-learning/ contains code, data, models and reports.\n'
                 'Retain every model together with its JSON/labels/preprocessing and optimizer state.\n'
                 'Recreate Python environments from environment.json; drivers/installed binaries/Git history excluded.\n'
                 'External Windows HDF5/recordings are included only if copied under the source root beforehand.\n'
                 'No automatic training resume or data upload.\n')
        for name,content in [('environment.json',json.dumps(environment,indent=2)),('RESTORE.txt',restore)]:
            raw=content.encode();z.writestr(name,raw);hashes[name]=hashlib.sha256(raw).hexdigest()
        z.writestr('SHA256SUMS.json',json.dumps(hashes,indent=2))
    count=verify_archive(partial)
    if set(collect(root))!=set(files) or any(signature(path)!=before[path] for path in files):
        raise ValueError('Source tree changed during backup; partial archive retained')
    archive_hash=digest(partial)
    partial.rename(archive)
    sidecar=archive.with_suffix('.zip.sha256')
    sidecar.write_text(archive_hash+'  '+archive.name+'\n',encoding='utf-8')
    print('Sicherung erfolgreich:',archive,flush=True)
    print('Gepruefte Dateien:',count,'| ZIP SHA256:',archive_hash,flush=True)
    return archive


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.home()/'rf-learning')
    p.add_argument('--output',type=Path,default=Path.home()/'rf-learning-backups')
    p.add_argument('--verify',type=Path)
    args=p.parse_args()
    try:
        if args.verify:print('ZIP geprueft:',verify_archive(args.verify),'Dateien')
        else:
            print('Bitte Training beenden und waehrend der Sicherung keine Quelldateien veraendern.',flush=True)
            create_backup(args.root,args.output)
    except (OSError,ValueError,KeyError,zipfile.BadZipFile,RuntimeError) as error:
        p.exit(1,'Sicherung fehlgeschlagen: '+str(error)+'\n')
