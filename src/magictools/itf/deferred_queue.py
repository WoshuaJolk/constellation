"""Append-only ingestion, bounded queue scans and supplemental warm starts."""
import hashlib,json,os
from collections import Counter


def key_for(record):
    inputs={k:record[k] for k in ('indices','initial','epoch','sigma')}
    inputs['indices']=sorted(set(map(int,inputs['indices'])))
    return hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()


def import_incremental(db,directories):
    db.execute('CREATE TABLE IF NOT EXISTS import_cursors(path TEXT PRIMARY KEY,device INTEGER,inode INTEGER,offset INTEGER,head_bytes INTEGER,head_sha TEXT)')
    counts=Counter()
    for directory in directories:
        for path in sorted((directory/'deferred-fits').glob('*.jsonl')):
            name=str(path.resolve());stat=path.stat()
            prior=db.execute('SELECT device,inode,offset,head_bytes,head_sha FROM import_cursors WHERE path=?',(name,)).fetchone()
            with path.open('rb') as stream:
                offset=0;head_bytes=min(4096,stat.st_size);head=hashlib.sha256(stream.read(head_bytes)).hexdigest()
                if prior:
                    stream.seek(0);previous_head=hashlib.sha256(stream.read(prior[3])).hexdigest()
                    if (stat.st_dev,stat.st_ino)==prior[:2] and stat.st_size>=prior[2] and previous_head==prior[4]:
                        offset=prior[2]
                        # Expand the fingerprint as an initially empty/short
                        # log grows, after checking its prior prefix.
                    else:counts['reset_sources']+=1
                stream.seek(offset);pending=0
                while stream.tell()<stat.st_size:
                    start=stream.tell();line=stream.readline()
                    if not line.endswith(b'\n') or stream.tell()>stat.st_size:break
                    record=json.loads(line);record['indices']=sorted(set(map(int,record['indices'])))
                    key=key_for(record)
                    changed=db.execute('INSERT OR IGNORE INTO jobs(key,body,status,attempts,error) VALUES (?,?,?,0,NULL)',(key,json.dumps(record),'pending')).rowcount
                    counts['new_jobs']+=changed;counts['duplicate_inputs']+=1-changed
                    counts['lines_read']+=1;counts['bytes_read']+=len(line);offset=stream.tell();pending+=1
                    if pending>=4096:
                        db.execute('INSERT OR REPLACE INTO import_cursors VALUES (?,?,?,?,?,?)',(name,stat.st_dev,stat.st_ino,offset,head_bytes,head));db.commit();pending=0
                db.execute('INSERT OR REPLACE INTO import_cursors VALUES (?,?,?,?,?,?)',(name,stat.st_dev,stat.st_ino,offset,head_bytes,head));db.commit()
                counts['sources']+=1
    return dict(counts)


def pending_jobs(db,max_jobs=0,batch=256):
    """Keyset pagination: bound memory, preserve attempt fairness and new jobs."""
    top=db.execute('SELECT max(attempts) FROM jobs').fetchone()[0] or 0
    supplied=0
    for attempt in range(top+1):
        key=''
        while True:
            rows=db.execute("SELECT key,body,attempts FROM jobs WHERE status IN ('pending','retryable') AND attempts=? AND key>? ORDER BY key LIMIT ?",(attempt,key,batch)).fetchall()
            if not rows:break
            for row in rows:
                yield row;supplied+=1
                if max_jobs and supplied>=max_jobs:return
            key=rows[-1][0]


class WarmBank:
    """Orbit initialization suggestions; never retire the original-start job."""
    def __init__(self,db):
        self.db=db
        db.executescript('CREATE TABLE IF NOT EXISTS warm_orbits(key TEXT PRIMARY KEY,body TEXT);'
            'CREATE TABLE IF NOT EXISTS warm_members(track INTEGER,key TEXT,PRIMARY KEY(track,key));'
            'CREATE TABLE IF NOT EXISTS warm_attempts(parent TEXT PRIMARY KEY,child TEXT);')

    def add(self,row):
        # Distinct modes remain separately addressable, without rounding.
        for mode in row.get('modes',[]):
            item=dict(indices=row['tracklet_indices'],initial=mode['state_au_au_per_day'],epoch=mode['epoch_mjd_tdb'],sigma=row.get('sigma_arcsec',.3),seed_key=f"{row.get('seed_rank',0)}:{row.get('seed_mode',0)}")
            key=key_for(item)
            if self.db.execute('INSERT OR IGNORE INTO warm_orbits VALUES (?,?)',(key,json.dumps(item))).rowcount:
                self.db.executemany('INSERT OR IGNORE INTO warm_members VALUES (?,?)',[(int(i),key) for i in item['indices']])

    def suggest(self,parent,record):
        if self.db.execute('SELECT 1 FROM warm_attempts WHERE parent=?',(parent,)).fetchone():return None
        indices=sorted(set(record['indices']));marks=','.join('?' for _ in indices)
        found=self.db.execute(f'SELECT key,count(*) AS shared FROM warm_members WHERE track IN ({marks}) GROUP BY key HAVING shared>=3 ORDER BY shared DESC,key LIMIT 8',indices).fetchall()
        for key,_ in found:
            start=json.loads(self.db.execute('SELECT body FROM warm_orbits WHERE key=?',(key,)).fetchone()[0])
            child=dict(record,initial=start['initial'],epoch=start['epoch'],warm_parent=parent,warm_source=key)
            child_key=key_for(child)
            if child_key==parent:continue
            inserted=self.db.execute('INSERT OR IGNORE INTO jobs(key,body,status,attempts,error) VALUES (?,?,?,0,NULL)',(child_key,json.dumps(child),'pending')).rowcount
            self.db.execute('INSERT OR IGNORE INTO warm_attempts VALUES (?,?)',(parent,child_key))
            return (child_key,child) if inserted else None
        return None
