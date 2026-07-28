#!/usr/bin/env python3
import csv, json, math, statistics, sys, hashlib, tarfile
from collections import defaultdict
from pathlib import Path

LABEL = {"A":"A (neutral)", "B":"B (rough_only_high)", "C":"C (fixed_high)"}
ACTIVE = ("flat","upslope","rough","downslope")

def f(v):
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None

def ms(vals):
    vals=[x for x in vals if x is not None]
    if not vals: return (None,None)
    return (statistics.mean(vals), statistics.stdev(vals) if len(vals)>1 else 0.0)

def fmt(x,n=3):
    return "NA" if x is None or not math.isfinite(x) else f"{x:.{n}f}"

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as fp:
        for b in iter(lambda: fp.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def write_csv(path, rows):
    keys=[]
    seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with open(path,"w",newline="",encoding="utf-8") as fp:
        w=csv.DictWriter(fp,fieldnames=keys); w.writeheader(); w.writerows(rows)

def read_rows(path):
    with open(path,newline="",encoding="utf-8") as fp:
        rows=list(csv.DictReader(fp))
    rows=[r for r in rows if r.get("valid")=="1" and f(r.get("sim_time")) is not None]
    if len(rows)<500: raise RuntimeError(f"{path}: only {len(rows)} valid rows")
    return rows

def integrate(rows, stop):
    total=defaultdict(float)
    seg=defaultdict(lambda: defaultdict(float))
    fields={"work_abs":"power_abs_W_cmd","work_pos":"power_positive_W_cmd",
            "tau2":"tau_sq_sum","dq2":"dq_sq_sum"}
    for i in range(1,stop+1):
        a,b=rows[i-1],rows[i]
        ta,tb=f(a["sim_time"]),f(b["sim_time"])
        dt=tb-ta
        if not (0<dt<=0.2): continue
        ca,cb=a.get("context","unknown"),b.get("context","unknown")
        total["duration"]+=dt
        if ca==cb: seg[ca]["duration"]+=dt
        else:
            seg[ca]["duration"]+=dt/2; seg[cb]["duration"]+=dt/2
        for out,src in fields.items():
            va,vb=f(a.get(src)),f(b.get(src))
            if va is None or vb is None: continue
            z=(va+vb)*0.5*dt
            total[out]+=z
            if ca==cb: seg[ca][out]+=z
            else:
                seg[ca][out]+=z/2; seg[cb][out]+=z/2
    return dict(total), {k:dict(v) for k,v in seg.items()}

def analyze(m):
    p=Path(m["log_dir"])/"ros1_joint_energy_proxy.csv"
    rows=read_rows(p)
    goal=next((i for i,r in enumerate(rows) if r.get("context")=="goal_flat"),None)
    if goal is None: raise RuntimeError(f"run {m['run_index']}: no goal_flat")
    pre=rows[:goal+1]
    t0,tg=f(pre[0]["sim_time"]),f(pre[-1]["sim_time"])
    xs=[f(r["x"]) for r in pre]; ys=[f(r["y"]) for r in pre]
    xs=[x for x in xs if x is not None]; ys=[y for y in ys if y is not None]
    total,seg=integrate(rows,goal)
    geom={}
    for c in set(r.get("context","unknown") for r in pre):
        rr=[r for r in pre if r.get("context","unknown")==c]
        yy=[f(r["y"]) for r in rr]; yy=[y for y in yy if y is not None]
        xx=[f(r["x"]) for r in rr]; xx=[x for x in xx if x is not None]
        geom[c]={"abs_dy":abs(yy[-1]-yy[0]) if len(yy)>1 else None,
                 "max_abs_y":max(abs(y) for y in yy) if yy else None,
                 "dx":xx[-1]-xx[0] if len(xx)>1 else None}
    segment=[]
    for c in sorted(set(seg)|set(geom)):
        d=seg.get(c,{})
        dur=d.get("duration")
        segment.append({
            "run_index":int(m["run_index"]),"block":int(m["block"]),
            "condition":m["condition"],"context":c,
            "duration_s":dur,"abs_dy":geom.get(c,{}).get("abs_dy"),
            "max_abs_y":geom.get(c,{}).get("max_abs_y"),
            "work_abs_J_proxy":d.get("work_abs"),
            "work_positive_J_proxy":d.get("work_pos"),
            "tau_sq_integral":d.get("tau2"),"dq_sq_integral":d.get("dq2"),
            "mean_power_abs_W_proxy":(d.get("work_abs")/dur if dur and d.get("work_abs") is not None else None)
        })
    byc={r["context"]:r for r in segment}
    rough=byc.get("rough",{})
    nonrough=sum((byc.get(c,{}).get("work_abs_J_proxy") or 0.0) for c in ("flat","upslope","downslope"))
    dist=xs[-1]-xs[0]
    run={
        "run_index":int(m["run_index"]),"block":int(m["block"]),
        "condition":m["condition"],"condition_name":m["condition_name"],
        "rc":int(m["rc"]),"log_dir":m["log_dir"],
        "goal_time_sim_s":tg-t0,"goal_entry_x":xs[-1],
        "goal_entry_y":ys[-1],"goal_entry_abs_y":abs(ys[-1]),
        "max_abs_y_pre_goal":max(abs(y) for y in ys),
        "rough_abs_dy":rough.get("abs_dy"),
        "goal_work_abs_J_proxy":total.get("work_abs"),
        "goal_work_positive_J_proxy":total.get("work_pos"),
        "goal_tau_sq_integral":total.get("tau2"),
        "goal_dq_sq_integral":total.get("dq2"),
        "work_abs_per_m_J_proxy":total.get("work_abs")/dist if dist>1e-9 else None,
        "rough_work_abs_J_proxy":rough.get("work_abs_J_proxy"),
        "nonrough_work_abs_J_proxy":nonrough,
    }
    return run,segment

def summarize(runs):
    metrics=("goal_time_sim_s","goal_entry_abs_y","max_abs_y_pre_goal","rough_abs_dy",
             "goal_work_abs_J_proxy","goal_work_positive_J_proxy","goal_tau_sq_integral",
             "goal_dq_sq_integral","work_abs_per_m_J_proxy","rough_work_abs_J_proxy",
             "nonrough_work_abs_J_proxy")
    out=[]
    for c in "ABC":
        rr=[r for r in runs if r["condition"]==c]
        row={"condition":c,"condition_label":LABEL[c],"n":len(rr)}
        for k in metrics:
            a,b=ms([f(r.get(k)) for r in rr]); row[k+"_mean"]=a; row[k+"_std"]=b
        out.append(row)
    return out

def paired(runs):
    metrics=("goal_time_sim_s","goal_entry_abs_y","max_abs_y_pre_goal","rough_abs_dy",
             "goal_work_abs_J_proxy","goal_work_positive_J_proxy","goal_tau_sq_integral",
             "goal_dq_sq_integral","work_abs_per_m_J_proxy","rough_work_abs_J_proxy",
             "nonrough_work_abs_J_proxy")
    blocks=defaultdict(dict)
    for r in runs: blocks[r["block"]][r["condition"]]=r
    out=[]
    for name,l,r in (("B_minus_A","B","A"),("C_minus_A","C","A"),("B_minus_C","B","C")):
        for k in metrics:
            e=[]; rel=[]
            for b in sorted(blocks):
                if l not in blocks[b] or r not in blocks[b]: continue
                x,y=f(blocks[b][l].get(k)),f(blocks[b][r].get(k))
                if x is None or y is None: continue
                e.append(x-y)
                if abs(y)>1e-12: rel.append(100*(x-y)/y)
            a,s=ms(e); ar,sr=ms(rel)
            out.append({"contrast":name,"metric":k,"n_blocks":len(e),
                        "mean_effect":a,"std_effect":s,
                        "mean_relative_effect_pct":ar,"std_relative_effect_pct":sr,
                        "positive_count":sum(x>0 for x in e),
                        "negative_count":sum(x<0 for x in e)})
    return out

def segment_summary(seg):
    out=[]
    for c in "ABC":
        for ctx in ACTIVE:
            rr=[r for r in seg if r["condition"]==c and r["context"]==ctx]
            row={"condition":c,"condition_label":LABEL[c],"context":ctx,"n":len(rr)}
            for k in ("duration_s","abs_dy","max_abs_y","work_abs_J_proxy",
                      "work_positive_J_proxy","tau_sq_integral","dq_sq_integral",
                      "mean_power_abs_W_proxy"):
                a,s=ms([f(r.get(k)) for r in rr]); row[k+"_mean"]=a; row[k+"_std"]=s
            out.append(row)
    return out

def markdown(runs,cond,pair,segsum):
    L=["# TRACER LL1 Clearance Energy A/B/C Balanced Analysis","",
       "## Collection validity","",
       f"- runs: {len(runs)}","- balanced conditions: A/B/C = 3/3/3",
       f"- all runs rc=0: {sum(r['rc']==0 for r in runs)}/{len(runs)}",
       "- energy window ends at first `goal_flat` entry; post-goal hold is excluded.",
       "- energy values are commanded mechanical-work/effort proxies, not battery electrical energy.","",
       "## Per-run outcome and energy","",
       "| run | block | condition | goal sim time | goal |y| | max |y| | rough |Δy| | work |τq̇| | positive work | τ² integral | work/m |",
       "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(runs,key=lambda x:x["run_index"]):
        L.append(f"| {r['run_index']} | {r['block']} | {LABEL[r['condition']]} | {fmt(r['goal_time_sim_s'])} | {fmt(r['goal_entry_abs_y'])} | {fmt(r['max_abs_y_pre_goal'])} | {fmt(r['rough_abs_dy'])} | {fmt(r['goal_work_abs_J_proxy'],1)} | {fmt(r['goal_work_positive_J_proxy'],1)} | {fmt(r['goal_tau_sq_integral'],1)} | {fmt(r['work_abs_per_m_J_proxy'],1)} |")
    L += ["","## Condition summary","",
          "| condition | goal time | goal |y| | max |y| | rough |Δy| | work |τq̇| | positive work | τ² integral | work/m |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in cond:
        L.append(f"| {r['condition_label']} | {fmt(r['goal_time_sim_s_mean'])} ± {fmt(r['goal_time_sim_s_std'])} | {fmt(r['goal_entry_abs_y_mean'])} ± {fmt(r['goal_entry_abs_y_std'])} | {fmt(r['max_abs_y_pre_goal_mean'])} ± {fmt(r['max_abs_y_pre_goal_std'])} | {fmt(r['rough_abs_dy_mean'])} ± {fmt(r['rough_abs_dy_std'])} | {fmt(r['goal_work_abs_J_proxy_mean'],1)} ± {fmt(r['goal_work_abs_J_proxy_std'],1)} | {fmt(r['goal_work_positive_J_proxy_mean'],1)} ± {fmt(r['goal_work_positive_J_proxy_std'],1)} | {fmt(r['goal_tau_sq_integral_mean'],1)} ± {fmt(r['goal_tau_sq_integral_std'],1)} | {fmt(r['work_abs_per_m_J_proxy_mean'],1)} ± {fmt(r['work_abs_per_m_J_proxy_std'],1)} |")
    L += ["","## Segment work summary","",
          "| condition | context | work |τq̇| mean±std | positive work mean±std | mean power |",
          "|---|---|---:|---:|---:|"]
    for r in segsum:
        L.append(f"| {r['condition_label']} | {r['context']} | {fmt(r['work_abs_J_proxy_mean'],1)} ± {fmt(r['work_abs_J_proxy_std'],1)} | {fmt(r['work_positive_J_proxy_mean'],1)} ± {fmt(r['work_positive_J_proxy_std'],1)} | {fmt(r['mean_power_abs_W_proxy_mean'],2)} ± {fmt(r['mean_power_abs_W_proxy_std'],2)} |")
    L += ["","## Paired block effects","",
          "Effects are left minus right. For time, lateral deviation, work, and effort, negative is preferable.","",
          "| contrast | metric | mean effect | std | relative effect | signs (+/-) |",
          "|---|---|---:|---:|---:|---:|"]
    for r in pair:
        L.append(f"| {r['contrast']} | {r['metric']} | {fmt(r['mean_effect'],4)} | {fmt(r['std_effect'],4)} | {fmt(r['mean_relative_effect_pct'],2)}% | {r['positive_count']}/{r['negative_count']} |")
    L += ["","## Interpretation guard","",
          "- n=3 per condition is balanced descriptive screening evidence, not final statistical confirmation.",
          "- B must preserve rough-terrain behavior while reducing non-rough work relative to C to support terrain-specific scheduling.",
          "- Similar B and C results support a generic high-clearance effect rather than Objective Selector value.",
          "- Large block-to-block sign reversals require additional paired repeats before promotion.",""]
    return "\n".join(L)

def main():
    if len(sys.argv)!=2: raise SystemExit(f"usage: {Path(sys.argv[0]).name} RUN_ROOT")
    root=Path(sys.argv[1]).expanduser().resolve()
    manifest=root/"manifest.tsv"
    with open(manifest,newline="",encoding="utf-8") as fp:
        mm=list(csv.DictReader(fp,delimiter="\t"))
    if len(mm)!=9: raise SystemExit(f"expected 9 manifest rows, found {len(mm)}")
    runs=[]; seg=[]
    for m in sorted(mm,key=lambda x:int(x["run_index"])):
        r,s=analyze(m); runs.append(r); seg.extend(s)
    cond=summarize(runs); pair=paired(runs); segsum=segment_summary(seg)
    out=root/"analysis_energy_v0"; out.mkdir(exist_ok=True)
    write_csv(out/"run_metrics_energy_v0.csv",runs)
    write_csv(out/"segment_metrics_energy_v0.csv",seg)
    write_csv(out/"condition_summary_energy_v0.csv",cond)
    write_csv(out/"segment_condition_summary_energy_v0.csv",segsum)
    write_csv(out/"paired_effects_energy_v0.csv",pair)
    payload={"run_root":str(root),"run_metrics":runs,"segment_metrics":seg,
             "condition_summary":cond,"segment_condition_summary":segsum,
             "paired_effects":pair}
    (out/"analysis_energy_v0.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    md=markdown(runs,cond,pair,segsum)
    (out/"summary_energy_v0.md").write_text(md+"\n",encoding="utf-8")
    print(md)
    files=[p for p in out.iterdir() if p.is_file()]
    (out/"SHA256SUMS").write_text("".join(f"{sha(p)}  {p.name}\n" for p in sorted(files)),encoding="utf-8")
    archive=root.parent/f"{root.name}_analysis_energy_v0.tar.gz"
    with tarfile.open(archive,"w:gz") as tar:
        tar.add(out,arcname=out.name); tar.add(manifest,arcname="manifest.tsv")
    print(f"\nanalysis_dir={out}\narchive={archive}\narchive_sha256={sha(archive)}")

if __name__=="__main__": main()
