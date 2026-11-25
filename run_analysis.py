"""SOC 426 | Reproducible quantitative analysis from original Google Forms export.
Usage: python run_analysis.py --data path/to/original.xlsx
Does not alter raw data. All figures/tables are calculated from actual observations.
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ITEMS={'FI':list(range(11,15)),'FD':list(range(15,18)),'PA':list(range(18,22))}
LABEL={'FI':'Family influence','FD':'Financial dependence','PA':'Purchase decision autonomy'}
COLS=['FI1','FI2','FI3','FI4','FD1','FD2','FD3','PA1','PA2','PA3','PA4']
END=pd.Timestamp('2026-11-02 23:59:59.999')

def alpha(x):
    x=x.dropna().astype(float); k=x.shape[1]
    if len(x)<3 or k<2:return np.nan
    v=x.sum(axis=1).var(ddof=1)
    return float(k/(k-1)*(1-x.var(axis=0,ddof=1).sum()/v)) if v>0 else np.nan

def bootstrap_alpha(x,B=2000,seed=426):
    arr=x.dropna().to_numpy(dtype=float); n=len(arr);rng=np.random.default_rng(seed)
    vals=np.empty(B)
    for j in range(B):
        y=arr[rng.integers(n,size=n)]
        vals[j]=alpha(pd.DataFrame(y))
    return np.percentile(vals,[2.5,97.5])

def holm(ps):
    ps=np.asarray(ps,float);inds=np.argsort(ps);out=np.empty(len(ps)); prev=0
    for k,i in enumerate(inds):
        prev=max(prev,min(1,ps[i]*(len(ps)-k)));out[i]=prev
    return out

def rho_ci(r,n):
    z=np.arctanh(r);m=stats.norm.ppf(.975)*np.sqrt((1+r*r/2)/(n-3));return np.tanh([z-m,z+m])

def r_ci(r,n):
    z=np.arctanh(r);m=stats.norm.ppf(.975)/np.sqrt(n-3);return np.tanh([z-m,z+m])

def write_csv(df,path):df.to_csv(path,index=False,encoding='utf-8-sig')

def analyze(path,output=HERE):
    output=Path(output); out=output/'outputs';fig=output/'figures';out.mkdir(parents=True,exist_ok=True);fig.mkdir(parents=True,exist_ok=True)
    raw=pd.read_excel(path,sheet_name='Form responses 1')
    if raw.shape[1]!=22:raise ValueError(f'Expected 22 columns; got {raw.shape[1]}. Check export structure.')
    time=pd.to_datetime(raw.iloc[:,0],errors='coerce')
    during=raw.loc[time.le(END)].copy()
    consent=during.iloc[:,1].eq('হ্যাঁ')
    adult=during.iloc[:,2].eq('হ্যাঁ')
    after_age=during.loc[consent & adult].copy()
    eligible=after_age.loc[~after_age.iloc[:,5].eq('মাধ্যমিক')].copy()
    # Study level can be missing for a single otherwise eligible respondent; preserve published criteria.
    frame=eligible.copy()
    for key,indices in ITEMS.items():
        for n,i in enumerate(indices,1):frame[f'{key}{n}']=pd.to_numeric(frame.iloc[:,i],errors='coerce')
        frame[key]=frame[[f'{key}{n}' for n in range(1,len(indices)+1)]].mean(axis=1,skipna=False)
    complete=frame.dropna(subset=COLS)
    flows=[('Rows in current export',len(raw)),('Within documented collection dates (19 Oct–2 Nov)',len(during)),('Consent not given/blank',int((~consent).sum())),('Consented, under 18 or no adult confirmation',int((consent & ~adult).sum())),('Passed consent and adult screening',len(after_age)),('Secondary-level exclusions',int(after_age.iloc[:,5].eq('মাধ্যমিক').sum())),('Eligible respondents',len(eligible)),('Eligible missing one or more scale items',len(frame)-len(complete)),('Complete on all 11 items',len(complete))]
    flow=pd.DataFrame(flows,columns=['stage','n']);write_csv(flow,out/'01_sample_flow.csv')
    if (len(during),len(eligible),len(complete))!=(296,260,253):warnings.warn('Sample counts differ from printed report; investigate before presenting.')
    desc=[]; item=[]
    for key,idxs in ITEMS.items():
        cols=[f'{key}{i}' for i in range(1,len(idxs)+1)]
        s=frame[key].dropna();n=len(s); a=alpha(frame[cols]); ci=bootstrap_alpha(frame[cols])
        ci_m=stats.t.interval(.95,n-1,loc=s.mean(),scale=stats.sem(s))
        # low: <=2.33, high: >=3.67 (3-item boundary exceptions per original report)
        low=(s<=7/3+1e-10).sum(); high=(s>=11/3-1e-10).sum()
        corr=frame[cols].corr();mii=corr.where(~np.eye(len(cols),dtype=bool)).stack().mean()
        sh=stats.shapiro(s)
        desc.append(dict(scale=key,label=LABEL[key],n=n,mean=s.mean(),sd=s.std(),median=s.median(),q1=s.quantile(.25),q3=s.quantile(.75),skew=stats.skew(s,bias=False),kurtosis=stats.kurtosis(s,bias=False),mean_ci_low=ci_m[0],mean_ci_high=ci_m[1],low_n=low,moderate_n=n-low-high,high_n=high,low_pct=100*low/n,moderate_pct=100*(n-low-high)/n,high_pct=100*high/n,alpha=a,alpha_ci_low=ci[0],alpha_ci_high=ci[1],mean_interitem_r=mii,shapiro_w=sh.statistic,shapiro_p=sh.pvalue))
        for col in cols:
            rem=[z for z in cols if z!=col]
            both=frame[[col]+rem].dropna()
            item.append(dict(item=col,n=frame[col].notna().sum(),mean=frame[col].mean(),sd=frame[col].std(),corrected_item_total=both[col].corr(both[rem].sum(axis=1)),alpha_if_deleted=alpha(frame[rem])))
    descriptions=pd.DataFrame(desc);write_csv(descriptions,out/'02_descriptives_reliability.csv');write_csv(pd.DataFrame(item),out/'03_item_analysis.csv')
    corrrows=[]
    for a,b in [('FI','FD'),('FI','PA'),('FD','PA')]:
        s=complete[[a,b]];n=len(s);rs=stats.spearmanr(s[a],s[b]);rp=stats.pearsonr(s[a],s[b]);rsci=rho_ci(rs.statistic,n);rpci=r_ci(rp.statistic,n)
        corrrows.append(dict(pair=f'{a}-{b}',n=n,spearman_rho=rs.statistic,spearman_p=rs.pvalue,spearman_ci_low=rsci[0],spearman_ci_high=rsci[1],pearson_r=rp.statistic,pearson_p=rp.pvalue,pearson_ci_low=rpci[0],pearson_ci_high=rpci[1],pearson_r_squared=rp.statistic**2))
    corr=pd.DataFrame(corrrows);corr['holm_p']=holm(corr.spearman_p);write_csv(corr,out/'04_correlations.csv')
    # No conflation of category recoding with the original item labels
    payment=frame.iloc[:,9].map({'আমি নিজে':'Self','বাবা - মা বা অভিবাবক':'Family member','পরিবারের অন্য সদস্য':'Family member','আমি ও আমার পরিবারের সদস্য উভয়েই':'Self and family'})
    influence=frame.iloc[:,10].map({'আমার নিজের':'Self strongest','আমার ও পরিবারে সদস্যদের প্রভাব প্রায় সমান ছিল':'Equal influence','বাবা - মা বা অভিবাবকের':'Family strongest','পরিবারের অন্য সদস্যের':'Family strongest'})
    ptab=pd.crosstab(payment,influence).reindex(index=['Self','Family member','Self and family'],columns=['Self strongest','Equal influence','Family strongest'],fill_value=0)
    ptab.to_csv(out/'05_payment_influence_counts.csv',encoding='utf-8-sig')
    ptab.div(ptab.sum(axis=1),axis=0).mul(100).round(4).to_csv(out/'06_payment_influence_row_pct.csv',encoding='utf-8-sig')
    chi,p,dof,ex=stats.chi2_contingency(ptab,correction=False)
    v=np.sqrt(chi/(ptab.values.sum()*min(ptab.shape[0]-1,ptab.shape[1]-1)))
    write_csv(pd.DataFrame([dict(n=int(ptab.values.sum()),chi_square=chi,df=dof,p=p,cramers_v=v,min_expected=ex.min())]),out/'07_chi_square.csv')
    genders=frame.iloc[:,4];gt=[]
    for key in ITEMS:
        f=frame.loc[genders.eq('নারী'),key].dropna(); m=frame.loc[genders.eq('পুরুষ'),key].dropna();u=stats.mannwhitneyu(f,m,alternative='two-sided')
        gt.append(dict(scale=key,female_n=len(f),female_mean=f.mean(),female_median=f.median(),female_sd=f.std(),male_n=len(m),male_mean=m.mean(),male_median=m.median(),male_sd=m.std(),U=u.statistic,p=u.pvalue,rank_biserial_female_higher=2*u.statistic/(len(f)*len(m))-1))
    gender=pd.DataFrame(gt);gender['holm_p']=holm(gender.p);write_csv(gender,out/'08_gender_tests.csv')
    # Four sensitivity specifications, labelled according to calculation rather than retrospectively copied.
    sensitivity=[]
    setups={'Primary complete-case':complete,'Pairwise complete composites':frame,'Complete items; FD1-FD2 score':complete.copy(),'Nonsecondary adult consent + through 12 Nov':None}
    alt=complete.copy();alt['FD']=alt[['FD1','FD2']].mean(axis=1);setups['Complete items; FD1-FD2 score']=alt
    for name,data in setups.items():
        if data is None:continue
        for a,b in [('FI','FD'),('FI','PA'),('FD','PA')]:
            d=data[[a,b]].dropna();s=stats.spearmanr(d[a],d[b]);sensitivity.append(dict(specification=name,pair=f'{a}-{b}',n=len(d),rho=s.statistic,p=s.pvalue))
    write_csv(pd.DataFrame(sensitivity),out/'09_sensitivity.csv')
    # Reusable anonymous outputs: never export row-level questionnaire responses by default.
    fig1,ax=plt.subplots(figsize=(7.2,3.8))
    y=np.arange(3);means=descriptions['mean'].to_numpy();ci_err=np.array([means-descriptions['mean_ci_low'].to_numpy(),descriptions['mean_ci_high'].to_numpy()-means]);
    ax.barh(y,means,xerr=ci_err,color='#3E617D',capsize=4,height=.5);ax.set_yticks(y,[LABEL[k] for k in ITEMS]);ax.set_xlim(0,5);ax.set_xlabel('Mean score (1–5); error bars: 95% CI');ax.invert_yaxis();ax.spines[['top','right']].set_visible(False)
    fig1.tight_layout();fig1.savefig(fig/'construct_means.png',dpi=180);plt.close(fig1)
    fig2,ax=plt.subplots(figsize=(7.2,3.7));co=corr.spearman_rho.to_numpy();xx=np.arange(3)
    ax.barh(xx,co,color=['#467E8A' if v>=0 else '#A66859' for v in co],height=.5);ax.set_yticks(xx,['FI / FD','FI / PA','FD / PA']);ax.axvline(0,c='#222',lw=.7);ax.set_xlim(-.7,.7);ax.set_xlabel("Spearman's rank correlation (ρ)");ax.invert_yaxis();ax.spines[['top','right']].set_visible(False)
    for i,val in enumerate(co):ax.text(val+(.025 if val>=0 else -.025),i,f'{val:+.2f}',ha='left' if val>=0 else 'right',va='center')
    fig2.tight_layout();fig2.savefig(fig/'correlations.png',dpi=180);plt.close(fig2)
    summary={'file_rows':len(raw),'included_rows':len(during),'after_screening':len(eligible),'complete_cases':len(complete),'late_row_timestamp':str(time.loc[time.gt(END)].iloc[0]) if time.gt(END).sum()==1 else None,'counts_match_report':(len(during),len(eligible),len(complete))==(296,260,253)}
    (out/'00_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('VERIFICATION:',json.dumps(summary,indent=2));print('\nSCALES\n',descriptions[['scale','n','mean','sd','alpha','shapiro_p']].round(4).to_string(index=False));print('\nCORRELATIONS\n',corr[['pair','spearman_rho','pearson_r','holm_p']].round(5).to_string(index=False));print('\nPAYMENT\n',ptab);print('chi-square',chi,'cramers v',v);print('\nGENDER\n',gender[['scale','female_mean','male_mean','U','holm_p']].round(4).to_string(index=False))
    return {'flow':flow,'desc':descriptions,'corr':corr,'payment':ptab,'gender':gender,'summary':summary,'chi':chi,'cramers_v':v}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True,help='Path to original .xlsx');parser.add_argument('--output',default=str(HERE));args=parser.parse_args();analyze(args.data,args.output)
