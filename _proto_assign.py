# -*- coding: utf-8 -*-
"""Prototype assignation : detecteur+lecteur -> reconstruction de grille -> ordre canonique.
Le lecteur sert d'ANCRE (lectures sures) ; l'ordre canonique assigne le reste par position."""
import os, glob, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, cv2, torch
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision.models import resnet18

LEADS=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']  # ordre canonique colonne-major
dev='cuda' if torch.cuda.is_available() else 'cpu'
det=torch.load('data/training/runs_label_detector/run_latest/best_label_detector.pth',map_location=dev,weights_only=False)
DW,DH=det.get('in_w',1024),det.get('in_h',736)
mdet=smp.Unet('resnet34',encoder_weights=None,in_channels=3,classes=1,activation=None).to(dev); mdet.load_state_dict(det['model_state_dict']); mdet.eval()
rd=torch.load('data/training/runs_label_reader/run_latest/best_label_reader.pth',map_location=dev,weights_only=False)
reader=resnet18(weights=None,num_classes=12).to(dev); reader.load_state_dict(rd['model_state_dict']); reader.eval()

def cluster1d(vals, tol):
    """Regroupe des valeurs 1D par ecart < tol. Retourne [(centre, [indices])]."""
    order=sorted(range(len(vals)), key=lambda i: vals[i])
    groups=[[order[0]]]
    for i in order[1:]:
        if vals[i]-vals[groups[-1][-1]]<=tol: groups[-1].append(i)
        else: groups.append([i])
    return [(float(np.mean([vals[i] for i in g])), g) for g in groups]

def detect_and_read(img, THRESH=0.5):
    H,W=img.shape[:2]
    sc=min(DW/W,DH/H); nw,nh=int(round(W*sc)),int(round(H*sc))
    cv0=np.full((DH,DW,3),255,np.uint8); ox,oy=(DW-nw)//2,(DH-nh)//2; cv0[oy:oy+nh,ox:ox+nw]=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)
    with torch.no_grad(): hm=torch.sigmoid(mdet(torch.from_numpy(cv0.astype(np.float32)/255).permute(2,0,1).unsqueeze(0).to(dev)))[0,0].cpu().numpy()
    d=cv2.dilate(hm,np.ones((17,17),np.float32)); ys,xs=np.where((hm>=d)&(hm>THRESH))
    rsc=W/float(rd.get('train_w',3648)); half=max(8,int(round(rd['crop_half']*rsc))); R=rd['model_res']
    out=[]
    for px,py in zip(xs,ys):
        cx,cy=(px-ox)/sc,(py-oy)/sc; cxi,cyi=int(cx),int(cy)
        c=img[max(0,cyi-half):cyi+half, max(0,cxi-half):cxi+half]
        if c.size==0: continue
        cc=cv2.resize(c,(R,R),interpolation=cv2.INTER_AREA); t=torch.from_numpy(cc.astype(np.float32)/255).permute(2,0,1).unsqueeze(0).to(dev)
        with torch.no_grad(): pr=torch.softmax(reader(t),1)[0].cpu().numpy()
        out.append({'x':cx,'y':cy,'hm':float(hm[py,px]),'lead':LEADS[int(pr.argmax())],'conf':float(pr.max())})
    return out, half

def assign(img, THRESH=0.5):
    H,W=img.shape[:2]
    dets,half=detect_and_read(img,THRESH)
    anchors=[d for d in dets if d['conf']>=0.90]
    if len(anchors)<3: return None
    # 1) COLONNES : cluster x. Une VRAIE colonne est PLEINE (~nrows membres) ; les colonnes
    # FANTOMES (FP sur QRS des longues bandes) sont CREUSES. -> garde celles a >= moitie de
    # la colonne la plus peuplee (vire les fantomes type id 19).
    colc=cluster1d([d['x'] for d in anchors], tol=0.07*W)
    maxpop=max(len(g) for _,g in colc)
    colx=sorted(cx for cx,g in colc if len(g)>=max(2, 0.5*maxpop))
    if not colx: return None
    # re-assigne chaque ancre a la colonne la plus proche (drop si trop loin = FP hors-grille)
    kept=[d for d in anchors if min(abs(d['x']-cx) for cx in colx)<0.07*W]
    for d in kept: d['col']=int(np.argmin([abs(d['x']-cx) for cx in colx]))
    # 2) RANGEES : cluster GLOBAL des y (les rangees sont partagees entre colonnes)
    rowc=cluster1d([d['y'] for d in kept], tol=0.045*H)
    def strong(idxs): return len(colx)<2 or len({kept[i]['col'] for i in idxs})>=2
    strong_rows=sorted(cy for cy,idxs in rowc if strong(idxs))
    if len(strong_rows)>=2:
        # grille reguliere estimee sur les rangees solides (multi-colonnes)
        spacing=float(np.median(np.diff(strong_rows))); y0=strong_rows[0]
        lo,hi=strong_rows[0]-0.5*spacing, strong_rows[-1]+0.5*spacing
        rows_ok=[]
        for cy,idxs in rowc:
            k=round((cy-y0)/spacing); on_grid=abs((cy-y0)-k*spacing)<0.35*spacing
            # solide -> garde ; on-grid -> garde SEULEMENT dans l'etendue des solides (pas d'extrapolation -> vire les FP au-dela)
            if strong(idxs) or (on_grid and lo<=cy<=hi): rows_ok.append(cy)
        rowy=sorted(rows_ok)
    else:
        rowy=sorted(c for c,_ in rowc)
    ncols,nrows=len(colx),len(rowy)
    # 3) GRILLE ncols x nrows : chaque cellule (ci,ri) en (colx[ci], rowy[ri]), + detection la plus proche
    cells=[]
    for ci in range(ncols):
        for ri in range(nrows):
            cand=[d for d in kept if d['col']==ci and abs(d['y']-rowy[ri])<0.045*H]
            det_here=max(cand,key=lambda d:d['hm']) if cand else None
            cells.append({'col':ci,'row':ri,'cx':colx[ci],'y':rowy[ri],'fi':ci*nrows+ri,'det':det_here})
    # 4) ORDRE CANONIQUE colonne-major, ANCRE sur les lectures fiables.
    # idx_canon(lead) doit = fi + offset. On estime offset = mediane sur les cellules lues.
    offs=[LEADS.index(g['det']['lead'])-g['fi'] for g in cells if g['det'] and g['det']['lead'] in LEADS]
    offset=int(round(np.median(offs))) if offs else 0
    for g in cells:
        idx=g['fi']+offset
        g['canon']=LEADS[idx] if 0<=idx<len(LEADS) else '?'
    return {'cells':cells,'half':half,'ncols':ncols,'nrows':nrows,'anchors':len(kept),'offset':offset}

def viz(img, res, name):
    H,W=img.shape[:2]; half=res['half']; disp=img.copy()
    agree=0; total_with_det=0
    for g in res['cells']:
        x,y=int(g['cx']),int(g['y']); canon=g['canon']
        det=g['det']; has=det is not None
        if has:
            total_with_det+=1
            ok=(det['lead']==canon); agree+=int(ok)
            col=(0,170,0) if ok else (255,140,0)   # vert=lecteur d'accord, orange=desaccord (on garde canon)
        else: col=(0,90,255)  # bleu = extrapole (pas de detection, assigne par position)
        cv2.rectangle(disp,(x-half,y-half),(x+half,y+half),col,4)
        dl = det['lead'] if has else ''
        txt = canon + ('/'+dl if (has and dl != canon) else '')
        cv2.putText(disp,txt,(x-half,y-half-10),cv2.FONT_HERSHEY_SIMPLEX,1.0,col,3)
    sp=1300/W; Image.fromarray(cv2.resize(disp,(int(W*sp),int(H*sp)))).save(f'stage/_proto/{name}.png')
    return agree,total_with_det

os.makedirs('stage/_proto',exist_ok=True)
for idn in [17,21,1,19]:
    cand=[]
    for sub in os.listdir('data/output_real'):
        p=glob.glob(os.path.join('data/output_real',sub,f'img_{idn}_page_0.*'))
        if p: cand.append(p[0])
    path=[c for c in cand if 'brightness_120' in c] or cand
    if not path: print(f'id {idn}: introuvable'); continue
    img=np.array(Image.open(path[0]).convert('RGB'))
    res=assign(img,0.5)
    if res is None: print(f'id {idn}: pas assez d ancres'); continue
    ag,tot=viz(img,res,f'assign_{idn}')
    cells=sorted(res['cells'],key=lambda g:(g['col'],g['row']))
    print(f"id {idn}: grille {res['ncols']}col x {res['nrows']}rangees, {len(cells)} cellules, {res['anchors']} ancres | accord lecteur/canon: {ag}/{tot}")
    print('   assignation canonique:', [g['canon'] for g in cells])
print('-> stage/_proto/assign_*.png')
