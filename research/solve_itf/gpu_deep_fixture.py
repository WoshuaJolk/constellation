"""Freeze an independent-ODE distant-object control catalogue with real background.

Simulated positions have no magnitude/detection-efficiency model. They cannot
establish survey completeness, discoveries, or the existence of a distant planet.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from astropy.time import Time
import erfa
from magictools.itf.catalogue import Catalogue,OBS_DTYPE,TRACK_DTYPE,unit_vectors
from magictools.itf.kernels import MU,C,ARCSEC,summaries


def prepare(source,output):
    output.mkdir(parents=True,exist_ok=False)
    cat=Catalogue(source)
    # Fixed calendar/ordinal selection, no orbital outcomes or injection labels.
    pick=np.flatnonzero((cat.tracks['t']>=60000)&(cat.tracks['t']<60010))[:2000]
    tracks=cat.tracks[pick].copy();obs=[];cursor=0
    for i,row in zip(pick,tracks):
        o=cat.obs[cat.tracks[i]['start']:cat.tracks[i]['start']+row['count']].copy()
        row['start']=cursor;cursor+=len(o);obs.append(o)
    rng=np.random.default_rng(20261028);truth=[];future=[]
    for c,distance in enumerate([8.,40.,80.,150.,300.,1000.]):
      for branch in range(2):
        e=.1 if branch==0 else .7;incl=np.deg2rad(15 if branch==0 else 120)
        nu=np.pi/3;h=np.sqrt(MU*distance*(1+e*np.cos(nu)))
        angle=.75*c+.3*branch
        R=np.array([[np.cos(angle),-np.sin(angle)*np.cos(incl),np.sin(angle)*np.sin(incl)],
                    [np.sin(angle),np.cos(angle)*np.cos(incl),-np.cos(angle)*np.sin(incl)],
                    [0,np.sin(incl),np.cos(incl)]])
        r=distance*np.array([np.cos(nu),np.sin(nu),0.]);vr=MU/h*e*np.sin(nu);vt=h/distance
        v=vr*r/distance+vt*np.array([-np.sin(nu),np.cos(nu),0.]);state=np.r_[R@r,R@v]
        utc_times=60000+np.repeat([0.,1.,3.,5.,7.,9.,365.,730.],4)+np.tile([0.,.015,.03,.045],8)
        utc=Time(utc_times,format='mjd',scale='utc');tdb=utc.tdb;epoch=float(tdb.mjd[0]);earth,_=erfa.epv00(tdb.jd1,tdb.jd2)
        def force(t,y):return np.r_[y[3:],-MU*y[:3]/np.linalg.norm(y[:3])**3]
        solutions=[solve_ivp(force,(0,800),state,method='DOP853',rtol=2e-12,atol=1e-14,dense_output=True),solve_ivp(force,(0,-30),state,method='DOP853',rtol=2e-12,atol=1e-14,dense_output=True)]
        assert all(s.success for s in solutions)
        vectors=[]
        for t,o in zip(tdb.mjd,earth['p']):
            emitted=t-epoch
            for _ in range(12):
                xyz=solutions[0 if emitted>=0 else 1].sol(emitted)[:3]-o
                emitted=t-epoch-np.linalg.norm(xyz)/C
            u=xyz/np.linalg.norm(xyz);east=np.array([-u[1],u[0],0.]);east/=np.linalg.norm(east);north=np.cross(u,east)
            u+=.1/ARCSEC*(rng.normal()*east+rng.normal()*north);u/=np.linalg.norm(u);vectors.append(u)
        vectors=np.array(vectors);ids=[]
        for night in range(8):
            o=np.empty(4,OBS_DTYPE);sl=slice(night*4,(night+1)*4);u=vectors[sl]
            o['t']=utc_times[sl];o['ra_mas']=np.rint((np.degrees(np.arctan2(u[:,1],u[:,0]))%360)*3600000).astype(np.int32)%1296000000
            o['dec_mas']=np.rint(np.degrees(np.arcsin(u[:,2]))*3600000).astype(np.int32)
            identifier=f'ZZ{c}{branch}{night}'.encode()
            if night>=6:
                future.append(dict(identity=f'{c}-{branch}',id=identifier.decode(),times_utc=o['t'].tolist(),ra_mas=o['ra_mas'].tolist(),dec_mas=o['dec_mas'].tolist(),station='500'));continue
            unit=unit_vectors(o['ra_mas']/3600000,o['dec_mas']/3600000);stats=summaries(o['t'],unit,np.array([0]),np.array([4]))[0]
            row=np.zeros(1,TRACK_DTYPE);row['id']=identifier;row['site']=b'500';row['start']=cursor;row['count']=4;row['t']=stats[0];row['u']=stats[1:4];row['v']=stats[4:7];row['rms']=stats[7];row['span']=stats[8]
            ids.append(identifier.decode());tracks=np.r_[tracks,row];obs.append(o);cursor+=4
        truth.append(dict(identity=f'{c}-{branch}',distance_au=distance,eccentricity=e,inclination_deg=float(np.rad2deg(incl)),state=state.tolist(),epoch_mjd_tdb=epoch,tracklet_ids=ids,noise_sigma_arcsec=.1))
    np.save(output/'tracklets.npy',tracks);np.save(output/'observations.npy',np.concatenate(obs))
    manifest=dict(schema=2,counts={'tracklets':len(tracks),'stored_observations':cursor},source_sha256=cat.manifest['source_sha256'],synthetic_controls=len(truth),real_background_tracklets=len(pick),selection='First up to 2000 catalogue ordinals with UTC mean time [60000,60010); synthetic controls appended without fitting-based selection',synthetic_warning='Independent two-body DOP853 controls with 0.1 arcsec noise; geocentric station500; no flux, detectability, planets or observational correlation model')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));(output/'truth.json').write_text(json.dumps(truth,indent=2));(output/'reserved.json').write_text(json.dumps(future,indent=2))
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()};(output/'hashes.json').write_text(json.dumps(files,indent=2));print(json.dumps(manifest))

def restore_fixture(archive,output):
    import shutil
    output.mkdir(parents=True,exist_ok=False)
    with np.load(archive,allow_pickle=False) as data:
        np.save(output/'tracklets.npy',data['tracklets'])
        np.save(output/'observations.npy',data['observations'])
    for name in ['manifest.json','truth.json','reserved.json','hashes.json']:
        shutil.copyfile(archive.parent/name,output/name)
    expected=json.loads((output/'hashes.json').read_text())
    for name,digest in expected.items():
        if hashlib.sha256((output/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Restored fixture hash mismatch: '+name)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--source',type=Path);g.add_argument('--archive',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();restore_fixture(a.archive,a.output) if a.archive else prepare(a.source,a.output)
