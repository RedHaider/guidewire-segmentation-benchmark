import argparse,json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter

def image(seed=27,size=96):
 r=np.random.default_rng(seed);mask=np.zeros((size,size),bool);xs=np.arange(8,size-8);ys=(size/2+18*np.sin(xs/15+r.uniform(-1,1))).astype(int);mask[ys,xs]=1;mask=gaussian_filter(mask.astype(float),.7)>.18;bg=gaussian_filter(r.normal(size=(size,size)),4);im=bg*1.1+gaussian_filter(mask.astype(float),1.0)*1.25+r.normal(scale=.22,size=(size,size));return im,mask
def dice(a,b):return float(2*np.sum(a&b)/(a.sum()+b.sum()+1e-9))
def evaluate(seed=27):
 scores={"global_threshold":[],"ridge_adaptive":[]}
 for i in range(60):
  im,m=image(seed+i);p1=im>np.quantile(im,.965);ridge=im-gaussian_filter(im,2.2);p2=ridge>np.quantile(ridge,.975);scores["global_threshold"].append(dice(m,p1));scores["ridge_adaptive"].append(dice(m,p2))
 return {k:{"mean_dice":float(np.mean(v)),"std_dice":float(np.std(v))} for k,v in scores.items()}
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
