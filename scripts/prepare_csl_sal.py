"""Audit the user's official split CSL archive; extract only a fixed development pair."""
import bisect
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/runs/20260907_csl_sal_smoke_v1'
PARTS = [Path.home() / 'Downloads' / ('cslhdemgsplit.zip.part' + s) for s in ('aa','ab','ac','ad','ae')]

class SplitReader(io.RawIOBase):
    """Seekable byte concatenation without duplicating the 12 GB download."""
    def __init__(self, paths):
        self.handles = [p.open('rb') for p in paths]
        self.ends = []; total = 0
        for p in paths:
            total += p.stat().st_size; self.ends.append(total)
        self.size = total; self.pos = 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        value = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if value < 0: raise ValueError('Negative seek')
        self.pos = value; return value
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        remaining = min(n, self.size - self.pos); chunks = []
        while remaining > 0:
            i = bisect.bisect_right(self.ends, self.pos)
            begin = self.ends[i-1] if i else 0
            self.handles[i].seek(self.pos-begin)
            block = self.handles[i].read(min(remaining, self.ends[i]-self.pos))
            if not block: raise EOFError('Unexpected end of split archive')
            chunks.append(block); self.pos += len(block); remaining -= len(block)
        return b''.join(chunks)
    def close(self):
        for f in self.handles: f.close()
        super().close()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()


def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--inspect',action='store_true'); args=ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with SplitReader(PARTS) as stream, zipfile.ZipFile(stream) as z:
        infos=z.infolist()
        metadata=[dict(name=i.filename, bytes=i.file_size, compressed_bytes=i.compress_size, crc32=f'{i.CRC:08x}') for i in infos]
        (OUT/'archive_members.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps(dict(parts=[dict(path=str(p),bytes=p.stat().st_size) for p in PARTS],members=len(infos),uncompressed_bytes=sum(i.file_size for i in infos),first_names=z.namelist()[:12])),flush=True)
        for info in infos:
            if 'readme' in info.filename.lower() and info.file_size < 100000:
                content=z.read(info).decode('utf8', errors='replace')
                (OUT/'dataset_readme.txt').write_text(content)
                print(content,flush=True)
        if args.inspect: return
        started=time.perf_counter()
        hashes=[]
        for p in PARTS:
            hashes.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
            print('Hashed',p.name,flush=True)
        (OUT/'download_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
        selected=[]; checked=[]
        dest=ROOT/'data/public/csl-hdemg'
        for number,info in enumerate(infos):
            name=Path(info.filename)
            assert not name.is_absolute() and '..' not in name.parts
            if info.is_dir(): continue
            keep = ('subject1' in name.parts and any(s in name.parts for s in ('session1','session2'))) or 'readme' in info.filename.lower()
            # Stream every member to exercise ZIP's CRC without decoding reserved signals.
            h=hashlib.sha256(); target=dest/name if keep else None
            if target:
                target.parent.mkdir(parents=True,exist_ok=True)
                assert not target.exists(), f'Refusing to overwrite {target}'
            output=target.open('wb') if target else None
            try:
                with z.open(info) as f:
                    while chunk:=f.read(8*1024*1024):
                        h.update(chunk)
                        if output: output.write(chunk)
            finally:
                if output: output.close()
            item=dict(name=info.filename,bytes=info.file_size,sha256=h.hexdigest(),crc_passed=True)
            checked.append(item)
            if target: selected.append(dict(**item,path=str(target)))
            if number%50==0: print('Checked member',number,'of',len(infos),flush=True)
        result=dict(scope='Byte integrity for all archives; only subject1 sessions1/2 extracted; other signals not numerically decoded',passed=True,members=checked,extracted=selected,wall_seconds=time.perf_counter()-started)
        (OUT/'archive_audit.json').write_text(json.dumps(result,indent=2)+'\n')
        print('Archive verified; extracted',len(selected),'members;',result['wall_seconds'],'seconds',flush=True)

if __name__=='__main__': main()
