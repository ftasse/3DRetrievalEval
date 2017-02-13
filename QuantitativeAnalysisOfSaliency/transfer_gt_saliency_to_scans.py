import os
import glob
from numpy.linalg import norm
from scipy.spatial import KDTree

import re
numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def get_basename(filename):
  return os.path.basename(os.path.splitext(filename)[0])

def readOFF(filename):
  with open(filename) as f:
    line = f.readline().strip()
    if (line != 'OFF'):
      return None
    line = f.readline()
    stats = [int(x) for x in line.split()]
    num_verts = stats[0]
    num_faces = stats[1]
    V = []
    F = []
    for i in range(0, num_verts):
      V.append([float(x) for x in f.readline().split()])
    for i in range(0, num_faces):
      F.append([int(x) for x in f.readline().split()][1:])
    # print V[0:2], F[0:2]
    return V, F

def processModel(model_filename, watertight_gt_dir, scans_models_dir, scans_gt_dir, num_scans=12):
  watertight_gt_raw_dir = watertight_gt_dir.replace('/Saliency', '/RawData')
  scans_gt_raw_dir = scans_gt_dir.replace('/Saliency', '/RawData')

  basename = get_basename(model_filename)
  gt_saliency = [float(x.strip()) for x in open(os.path.join(watertight_gt_dir, '%s.val'%basename))]
  gt_keypoints = []
  for k in range(30):
    kpt_filename = os.path.join(watertight_gt_raw_dir,'%s_%d.pid'%(basename,k))
    if not os.path.exists(kpt_filename):
      break
    gt_keypoints.append(set([int(x.strip()) for x in open(kpt_filename)]))

  V,F = readOFF(model_filename)
  tree = KDTree(V)
  for k in range(num_scans):
    print 'Process', os.path.join(scans_models_dir, '%s_%d.off'%(basename,k))
    Vs,Fs = readOFF(os.path.join(scans_models_dir, '%s_%d.off'%(basename,k)))
    scan_gt_saliency = [0 for v in Vs]
    scans_gt_raw = [[] for x in gt_keypoints]
    scans_gt_raw_ref = [set([]) for x in gt_keypoints]
    for i in range(len(Vs)):
      v = Vs[i]
      closest_id = tree.query([v], 1)[1][0]
      if (closest_id < 0 or closest_id >= len(gt_saliency)):
        print 'invalid', v, closest_id
      scan_gt_saliency[i] = gt_saliency[closest_id]
      for parc in range(len(gt_keypoints)):
        if closest_id in gt_keypoints[parc] and not closest_id in scans_gt_raw_ref[parc]:
          scans_gt_raw[parc].append(i)
          scans_gt_raw_ref[parc].add(closest_id)
    # open(os.path.join(scans_gt_dir, '%s_%d.val'%(basename,k)), 'w').write('\n'.join([str(x) for x in scan_gt_saliency]))
    for parc in range(len(gt_keypoints)):
      open(os.path.join(scans_gt_raw_dir, '%s_%d_%d.pid'%(basename,k,parc)), 'w').write('\n'.join([str(x) for x in scans_gt_raw[parc]]))


watertight_gt_dir = '/Users/floratasse/VMShare/projects/saliency+retrieval_repo/data/watertight_track_retrieval_data/GS/Saliency'
watertight_models_dir = '/Users/floratasse/VMShare/projects/saliency+retrieval_repo/data/watertight_track/'
scans_gt_dir = '/Users/floratasse/VMShare/projects/saliency+retrieval_repo/data/watertight_track_scans_retrieval_data/GS/Saliency'
scans_models_dir = '/Users/floratasse/VMShare/projects/saliency+retrieval_repo/data/watertight_track_scans'
watertight_models = sorted(glob.glob(os.path.join(watertight_models_dir, "*.off")), key=numericalSort)

for m in watertight_models:
  print m
  if (int(m.split('/')[-1].split('.')[0]) >= 0):
    processModel(m, watertight_gt_dir, scans_models_dir, scans_gt_dir)
  # break

