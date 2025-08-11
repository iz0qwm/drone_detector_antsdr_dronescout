import yaml, csv, glob, os
from pathlib import Path

base = Path('/home/raffaello/dataset/yolo_vision')
cfg = yaml.safe_load(open(base/'data.yaml'))
classes = cfg['names'] if 'names' in cfg else cfg['nc']  # adatta se serve

def walk(split):
  img_dir = base/split/images
  lbl_dir = base/split/labels
  rows=[]
  for lbl_path in glob.glob(str(lbl_dir/'*.txt')):
    with open(lbl_path) as f:
      for line in f:
        cid, x, y, w, h = line.strip().split()
        rows.append([os.path.splitext(os.path.basename(lbl_path))[0]+'.jpg', classes[int(cid)], x,y,w,h, split])
  return rows

rows = walk('train') + walk('valid') + walk('test')
with open(base/'yolo_annotations.csv','w', newline='') as f:
  w=csv.writer(f); w.writerow(['image','class','x','y','w','h','split']); w.writerows(rows)
print('Scritto:', base/'yolo_annotations.csv')

