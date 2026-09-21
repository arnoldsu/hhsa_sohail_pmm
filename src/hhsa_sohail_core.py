#!/usr/bin/env python3
"""Reusable Sohail-style dual-network HHSA experiment for any 1-D index."""
from __future__ import annotations
import json, os, random, shutil, sys, threading, time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
PACKAGE_ROOT=Path(__file__).resolve().parents[1]; ROOT=PACKAGE_ROOT; SERIES='index'
sys.path.insert(0,'/g/data/p66/ars599/HHSA_WK/hhsa-python/src')
import numpy as np, pandas as pd, psutil, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from hhsa import decompose
from hhsa.instantaneous import direct_quadrature

CFG=json.loads((PACKAGE_ROOT/'config.json').read_text()); SEED=CFG['seed']; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)
for d in ('data','results','figures','models'): (ROOT/d).mkdir(parents=True,exist_ok=True)
LEADS=CFG['leads']; H=CFG['history']; K=max(LEADS); RIDX=np.array(LEADS)-1; T0=time.time(); RES=[]; STOP=False
def monitor():
 p=psutil.Process(); p.cpu_percent(None)
 while not STOP:
  try: RES.append({'elapsed_s':time.time()-T0,'cpu_percent':p.cpu_percent(None),'rss_mib':p.memory_info().rss/2**20})
  except psutil.Error: pass
  time.sleep(1)
class MLP(nn.Module):
 def __init__(self,nin,nout,hidden=48,residual=False):
  super().__init__(); self.residual=residual; self.inp=nn.Linear(nin,hidden); self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden*2),nn.GELU(),nn.Linear(hidden*2,hidden)) for _ in range(2 if residual else 0)]); self.out=nn.Linear(hidden,nout)
 def forward(self,x):
  x=torch.nn.functional.gelu(self.inp(x))
  for b in self.blocks: x=torch.nn.functional.gelu(x+b(x))
  return self.out(x)
def fit(X,Y,tr,va,seed,nout=None,residual=False):
 torch.manual_seed(seed); mu=X[tr].mean(0); sd=np.maximum(X[tr].std(0),1e-7); ym=Y[tr].mean(0); ys=np.maximum(Y[tr].std(0),1e-7)
 xs=(X-mu)/sd; yy=(Y-ym)/ys; model=MLP(X.shape[1],Y.shape[1] if nout is None else nout,CFG['hidden'],residual); opt=torch.optim.AdamW(model.parameters(),2e-3,weight_decay=2e-4)
 dl=DataLoader(TensorDataset(torch.tensor(xs[tr],dtype=torch.float32),torch.tensor(yy[tr],dtype=torch.float32)),CFG['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed))
 xv=torch.tensor(xs[va],dtype=torch.float32); yv=torch.tensor(yy[va],dtype=torch.float32); best=None; bl=np.inf; stale=0
 for _ in range(CFG['epochs']):
  model.train()
  for bx,by in dl: opt.zero_grad(); loss=((model(bx)-by)**2).mean(); loss.backward(); opt.step()
  model.eval()
  with torch.no_grad(): vl=float(((model(xv)-yv)**2).mean())
  if vl<bl-1e-6: bl=vl; best={k:v.clone() for k,v in model.state_dict().items()}; stale=0
  else: stale+=1
  if stale>=CFG['patience']: break
 model.load_state_dict(best); return model,mu,sd,ym,ys
def pred(bundle,X):
 m,mu,sd,ym,ys=bundle; m.eval()
 with torch.no_grad(): return m(torch.tensor((X-mu)/sd,dtype=torch.float32)).numpy()*ys+ym
def met(obs,fc,method,lead,level='signal',series=None):
 obs=np.asarray(obs); fc=np.asarray(fc); corr=np.corrcoef(obs,fc)[0,1] if np.std(fc)>0 else np.nan
 return {'level':level,'series':SERIES if series is None else series,'method':method,'lead_months':lead,'n':len(obs),'correlation':corr,'rmse':np.sqrt(mean_squared_error(obs,fc)),'mae':mean_absolute_error(obs,fc),'r2':r2_score(obs,fc),'slope':np.polyfit(obs,fc,1)[0],'variance_ratio':np.var(fc)/np.var(obs)}
def period(z):
 q=np.flatnonzero(np.diff(np.signbit(z-np.mean(z)))); return float(2*np.median(np.diff(q))) if len(q)>2 else 60.
def forecast_hhsa_sohail(dates,y,name,output_dir,forecast_end,config=None):
 """Forecast every final method (M0/M1/M2/M4/M5) to a calendar end date."""
 global ROOT,SERIES,CFG,SEED,H,K,LEADS,RIDX
 ROOT=Path(output_dir); SERIES=str(name); CFG={**json.loads((PACKAGE_ROOT/'config.json').read_text()),**(config or {})}; SEED=CFG['seed']; H=CFG['history']
 dates=np.asarray(dates); y=np.asarray(y,dtype=float).reshape(-1); n=len(y); last=pd.Timestamp(dates[-1]); future=pd.date_range(last+pd.offsets.MonthBegin(1),pd.Timestamp(forecast_end),freq='MS'); K=len(future)
 if K<1: raise ValueError('forecast_end must be after the last observation')
 r=decompose(y,CFG['sample_rate'],max_imfs=CFG['max_imfs'],max_modulation_imfs=CFG['max_am_imfs'],mask_order=0,mask_order2=0,upsample_level=0); c=r.IMF.shape[1]-1
 freq,_,phase=direct_quadrature(r.IMF[:,:c],CFG['sample_rate']); ph=np.stack([np.cos(phase),np.sin(phase)],-1); ia=np.abs(r.am[:,:c]); parts=r.IMF2[:,:,:c]; residual=r.IMF[:,-1]
 origins=np.arange(H,n-K); tr=origins<int(.70*n); va=(origins>=int(.70*n))&(origins<int(.85*n)); Y=np.stack([y[o:o+K] for o in origins])
 # M0, M1 and M2 use the same targets and chronological split.
 m0=np.repeat(y[-1],K); raw=np.stack([y[o-H:o] for o in origins]); m1b=fit(raw,Y,tr,va,21000); m1=pred(m1b,y[-H:][None])[0]; torch.save(m1b[0].state_dict(),ROOT/'models/future_M1_direct_nn.pt')
 state=np.column_stack([y,r.IMF[:,:c],ia,freq,ph.reshape(n,-1),parts.reshape(n,-1),residual]); SX=state[origins-1]; m2b=fit(SX,Y,tr,va,22000,residual=True); m2=pred(m2b,state[-1:])[0]; torch.save(m2b[0].state_dict(),ROOT/'models/future_M2_hhsa_single_resnet.pt')
 # M3 event and component-strength heads; their physical synthesis is M4.
 ets=np.column_stack([r.IMF[:,:c],freq,ph.reshape(n,-1)]); EX=np.stack([ets[o-H:o].reshape(-1) for o in origins]); EY=np.stack([ph[o:o+K].transpose(1,0,2).reshape(-1) for o in origins]); eb=fit(EX,EY,tr,va,23000); ep_all=pred(eb,EX).reshape(len(origins),c,K,2); ep_all/=np.maximum(np.linalg.norm(ep_all,axis=-1,keepdims=True),1e-8); ep=pred(eb,ets[-H:].reshape(1,-1)).reshape(c,K,2); ep/=np.maximum(np.linalg.norm(ep,axis=-1,keepdims=True),1e-8); torch.save(eb[0].state_dict(),ROOT/'models/future_M3_event_nn.pt')
 amp_all=np.zeros((len(origins),c,K)); ahat=np.zeros((c,K))
 for i in range(c):
  for j in range(parts.shape[1]):
   z=parts[:,j,i]
   if np.std(z[:int(.70*n)])<1e-9: continue
   L=int(np.clip(2*period(z[:int(.70*n)]),12,H)); X=np.stack([z[o-L:o] for o in origins]); Z=np.stack([z[o:o+K] for o in origins]); bb=fit(X,Z,tr,va,23100+i*20+j); amp_all[:,i]+=pred(bb,X); ahat[i]+=pred(bb,z[-L:][None])[0]; torch.save(bb[0].state_dict(),ROOT/f'models/future_M3_strength_IMF{i+1}_AMIMF{j+1}.pt')
 RX=np.stack([residual[o-H:o] for o in origins]); RY=np.stack([residual[o:o+K] for o in origins]); rb=fit(RX,RY,tr,va,23900); rp_all=pred(rb,RX); rr=pred(rb,residual[-H:][None])[0]; torch.save(rb[0].state_dict(),ROOT/'models/future_M3_residual_nn.pt')
 carrier=ep[:,:,0]; imfhat=ahat*carrier; m4=imfhat.sum(0)+rr
 # M5 learns the final combination only from the later chronological block.
 FX=np.concatenate([amp_all.reshape(len(origins),-1),ep_all.reshape(len(origins),-1),rp_all],1); latest=np.concatenate([ahat.ravel(),ep.ravel(),rr])[None]; idx=np.flatnonzero(va); ftr=np.zeros(len(origins),bool); fva=np.zeros(len(origins),bool); cut=max(1,int(.8*len(idx))); ftr[idx[:cut]]=True; fva[idx[cut:]]=True; m5b=fit(FX,Y,ftr,fva,24000,residual=True); m5=pred(m5b,latest)[0]; torch.save(m5b[0].state_dict(),ROOT/'models/future_M5_fusion_resnet.pt')
 methods={'M0 Persistence':m0,'M1 Direct NN':m1,'M2 HHSA single ResNet':m2,'M4 Dual physical reconstruction':m4,'M5 Dual fusion ResNet':m5}; out=pd.DataFrame({'date':future,'lead_months':np.arange(1,K+1),**methods,'M4 residual':rr})
 for i in range(c): out[f'M4 IA IMF{i+1}']=ahat[i]; out[f'M4 carrier IMF{i+1}']=carrier[i]; out[f'M4 IMF{i+1}']=imfhat[i]
 out.to_csv(ROOT/'results/future_forecast_all_methods.csv',index=False)
 long=out.melt(id_vars=['date','lead_months'],value_vars=list(methods),var_name='method',value_name='forecast'); long.to_csv(ROOT/'results/future_forecast.csv',index=False)
 fig,ax=plt.subplots(figsize=(14,6)); ax.plot(pd.to_datetime(dates)[-120:],y[-120:],'k',lw=1.2,label='Observed'); colors=['.5','tab:blue','tab:orange','tab:red','tab:purple']
 for (label,z),color in zip(methods.items(),colors): ax.plot(future,z,lw=1.5,label=label,color=color)
 ax.axvline(last,color='.35',ls='--',label='Forecast origin'); ax.axhline(0,color='.5',lw=.7); ax.set(title=f'{name}: HHSA-Sohail all-method forecast to {future[-1]:%Y-%m}',ylabel=name); ax.grid(alpha=.2); ax.legend(fontsize=8,ncol=2); fig.tight_layout(); fig.savefig(ROOT/'figures/03_future_forecast_all_methods.png',dpi=220); plt.close(fig)
 meta={'series':str(name),'last_observation':str(last.date()),'forecast_start':str(future[0].date()),'forecast_end':str(future[-1].date()),'horizon_months':K,'methods':list(methods),'M3_role':'intermediate event/strength branches used by M4 and M5','warning':'Experimental offline/full-record decomposition; not real-time causal validation.'}; (ROOT/'results/future_forecast_metadata.json').write_text(json.dumps(meta,indent=2)+'\n'); return out

def run_hhsa_sohail(time_values,index,name,output_dir,config=None,forecast_end=None):
 global STOP,ROOT,SERIES,CFG,SEED,T0,RES,LEADS,H,K,RIDX
 ROOT=Path(output_dir); SERIES=str(name); CFG={**json.loads((PACKAGE_ROOT/'config.json').read_text()),**(config or {})}; SEED=CFG['seed']; LEADS=CFG['leads']; H=CFG['history']; K=max(LEADS); RIDX=np.array(LEADS)-1; T0=time.time(); RES=[]; STOP=False
 random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
 for folder in ('data','results','figures','models'): (ROOT/folder).mkdir(parents=True,exist_ok=True)
 th=threading.Thread(target=monitor,daemon=True); th.start()
 dates=np.asarray(time_values); y=np.asarray(index,dtype=float).reshape(-1)
 if len(dates)!=len(y): raise ValueError('time and index lengths differ')
 if not np.isfinite(y).all(): raise ValueError(f'{name}: index contains NaN or infinity')
 n=len(y); pd.DataFrame({'date':dates,str(name):y}).to_csv(ROOT/'data/input.csv',index=False)
 r=decompose(y,CFG['sample_rate'],max_imfs=CFG['max_imfs'],max_modulation_imfs=CFG['max_am_imfs'],mask_order=0,mask_order2=0,upsample_level=0); c=r.IMF.shape[1]-1
 freq,_,phase=direct_quadrature(r.IMF[:,:c],CFG['sample_rate']); ph=np.stack([np.cos(phase),np.sin(phase)],-1); ia=np.abs(r.am[:,:c]); parts=r.IMF2[:,:,:c]; residual=r.IMF[:,-1]
 origins=np.arange(H,n-K); tr=origins<int(.70*n); va=(origins>=int(.70*n))&(origins<int(.85*n)); te=origins>=int(.85*n)
 target=np.stack([y[o:o+K] for o in origins])[:,RIDX]
 # Common raw and HHSA state predictors.
 raw=np.stack([y[o-H:o] for o in origins]); state=np.column_stack([y,r.IMF[:,:c],ia,freq,ph.reshape(n,-1),parts.reshape(n,-1),residual]); stateX=state[origins-1]
 p0=np.repeat(raw[te,-1,None],len(LEADS),1)
 m1=fit(raw,target,tr,va,100); p1=pred(m1,raw[te]); torch.save(m1[0].state_dict(),ROOT/'models/M1_direct_nn.pt')
 m2=fit(stateX,target,tr,va,200,residual=True); p2=pred(m2,stateX[te]); torch.save(m2[0].state_dict(),ROOT/'models/M2_hhsa_single_resnet.pt')
 # Event branch: histories of IMF, IF and phasor; target future phasor.
 event_ts=np.column_stack([r.IMF[:,:c],freq,ph.reshape(n,-1)]); eventX=np.stack([event_ts[o-H:o].reshape(-1) for o in origins]); eventY=np.stack([ph[o:o+K][:, :, :].transpose(1,0,2)[:,RIDX,:].reshape(-1) for o in origins])
 em=fit(eventX,eventY,tr,va,300); ep_all=pred(em,eventX); ep=ep_all.reshape(len(origins),c,len(LEADS),2); ep/=np.maximum(np.linalg.norm(ep,axis=-1,keepdims=True),1e-8); torch.save(em[0].state_dict(),ROOT/'models/M3_event_nn.pt')
 # Strength branch: one small NN for every real AM component; adaptive history.
 amp_pred=np.zeros((len(origins),c,len(LEADS))); component_rows=[]; histories={}; comp_predictions=[]
 for i in range(c):
  for j in range(parts.shape[1]):
   z=parts[:,j,i]
   if np.std(z[:int(.70*n)])<1e-9: continue
   L=int(np.clip(2*period(z[:int(.70*n)]),12,H)); histories[f'IMF{i+1}_AMIMF{j+1}']=L
   X=np.stack([z[o-L:o] for o in origins]); Y=np.stack([z[o:o+K][RIDX] for o in origins]); bm=fit(X,Y,tr,va,400+i*20+j); pp=pred(bm,X); amp_pred[:,i]+=pp; comp_predictions.append(pp)
   torch.save(bm[0].state_dict(),ROOT/f'models/M3_strength_IMF{i+1}_AMIMF{j+1}.pt')
   for q,l in enumerate(LEADS): component_rows.append(met(Y[te,q],pp[te,q],'Strength component NN',l,'AM-IMF',f'IMF{i+1}_AMIMF{j+1}'))
 # First-level residual prediction is a separate slow-strength head.
 RX=np.stack([residual[o-H:o] for o in origins]); RY=np.stack([residual[o:o+K][RIDX] for o in origins]); rm=fit(RX,RY,tr,va,900); rp=pred(rm,RX); torch.save(rm[0].state_dict(),ROOT/'models/M3_residual_nn.pt')
 # Branch diagnostics.
 branch=[]
 truep=np.stack([ph[o:o+K].transpose(1,0,2)[:,RIDX,:] for o in origins]); truea=np.stack([ia[o:o+K].transpose(1,0)[:,RIDX] for o in origins])
 for q,l in enumerate(LEADS):
  dot=np.sum(ep[te,:,q]*truep[te,:,q],axis=-1).clip(-1,1); branch.append({'branch':'event','lead_months':l,'circular_mae_rad':float(np.mean(np.arccos(dot))),'phasor_cosine':float(np.mean(dot))})
  for i in range(c): branch.append({'branch':'strength','series':f'IMF{i+1}_IA','lead_months':l,**{k:v for k,v in met(truea[te,i,q],amp_pred[te,i,q],'Strength NN',l).items() if k in ('correlation','rmse','mae','r2')}})
 # M4 physical reconstruction.
 p4=(amp_pred*ep[:,:,:,0]).sum(1)+rp
 # M5 fusion trains only on validation base outputs, then evaluates test.
 fusionX=np.concatenate([amp_pred.reshape(len(origins),-1),ep.reshape(len(origins),-1),rp],1); ftr=va.copy(); vidx=np.flatnonzero(va); fva=np.zeros(len(origins),bool); fva[vidx[int(.8*len(vidx)):]]=True; ftr[fva]=False
 fm=fit(fusionX,target,ftr,fva,1000,residual=True); p5=pred(fm,fusionX[te]); torch.save(fm[0].state_dict(),ROOT/'models/M5_fusion_resnet.pt')
 methods={'M0 Persistence':p0,'M1 Direct NN':p1,'M2 HHSA single ResNet':p2,'M4 Dual physical reconstruction':p4[te],'M5 Dual fusion ResNet':p5}; rows=[]; prows=[]
 for name,p in methods.items():
  for q,l in enumerate(LEADS):
   rows.append(met(target[te,q],p[:,q],name,l))
   for k,o in enumerate(origins[te]): prows.append({'method':name,'origin_index':o,'target_index':o+l-1,'lead_months':l,'observed':target[te,q][k],'predicted':p[k,q]})
 pd.DataFrame(rows+component_rows).to_csv(ROOT/'results/metrics.csv',index=False); pd.DataFrame(prows).to_csv(ROOT/'results/predictions.csv',index=False); pd.DataFrame(branch).to_csv(ROOT/'results/branch_metrics.csv',index=False)
 pd.DataFrame({'component':histories.keys(),'history_months':histories.values()}).to_csv(ROOT/'results/strength_component_histories.csv',index=False)
 np.savez_compressed(ROOT/'results/hhsa_arrays.npz',date=dates,IMF=r.IMF,IA=ia,IF=freq,phase=phase,IMF2=parts)
 # Figures.
 mm=pd.DataFrame(rows); fig,axs=plt.subplots(1,3,figsize=(14,4))
 for name,g in mm.groupby('method'):
  for ax,key in zip(axs,['correlation','rmse','r2']): ax.plot(g.lead_months,g[key],'o-',label=name); ax.set(title=key,xlabel='Lead months'); ax.grid(alpha=.2)
 axs[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(ROOT/'figures/01_M0_M5_skill.png',dpi=220); plt.close(fig)
 fig,axs=plt.subplots(2,3,figsize=(15,8),sharex=True); axs=axs.flat
 pp=pd.DataFrame(prows)
 for ax,l in zip(axs,LEADS):
  g=pp[pp.lead_months==l]; obs=g.groupby('target_index')['observed'].first(); ax.plot(dates[obs.index.astype(int)],obs,'k',label='Observed')
  for name,h in g.groupby('method'): ax.plot(dates[h.target_index.astype(int)],h.predicted,lw=.8,label=name)
  ax.set_title(f'{l}-month'); ax.grid(alpha=.2)
 axs[0].legend(fontsize=5); axs[-1].axis('off'); fig.tight_layout(); fig.savefig(ROOT/'figures/02_test_forecasts.png',dpi=220); plt.close(fig)
 STOP=True; th.join(2); rr=pd.DataFrame(RES); rr.to_csv(ROOT/'results/resource_usage.csv',index=False); summary={'series':SERIES,'wall_s':time.time()-T0,'peak_rss_mib':float(rr.rss_mib.max()),'mean_cpu_percent':float(rr.cpu_percent.mean()),'imf_carriers':c,'residual_components':1,'offline_full_record':True}; (ROOT/'results/resource_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 audit={'label':'OFFLINE / DIAGNOSTIC — NOT REAL-TIME','full_record_decomposition':True,'chronological_split':True,'scalers_training_only':True,'random_time_shuffle':False,'M5_fusion_training':'validation-period base predictions only','nan_input_count':int(np.isnan(state).sum())}; (ROOT/'results/leakage_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 future=None
 if forecast_end is not None: future=forecast_hhsa_sohail(dates,y,SERIES,ROOT,forecast_end,CFG)
 print(mm.to_string(index=False)); print(json.dumps(summary)); return {'metrics':mm,'summary':summary,'future':future,'output_dir':ROOT}

