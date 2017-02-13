import pprint
import glob
import sys
import os.path
import subprocess
import numpy as np
from  numpy.linalg import norm
from scipy import stats
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, pchip
from scipy.spatial import KDTree
from scipy.io import savemat
from scipy.spatial.distance import pdist, squareform

import scipy.io
import pickle
import itertools
import random
import inspect

import re
numbers = re.compile(r'(\d+)')

try:
  import geodesic
except ImportError:
  print "Could not import geodesic. Build it on your platform. Ex: cd geodesic && python setup.py build_ext --inplace"


default_num_bins = 100
my_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Ref: https://github.com/cvzoya/saliency/tree/master/code_forMetrics

# Compute AUC, NSS and EMD

def read_saliency_map(filename, num=0):
  if os.path.exists(filename):
    vals = [float(x) for x in open(filename).readlines()]
  else:
    vals = []
  if len(vals) == 0:
    if num > 0:
      vals = [0 for i in range(num)]
    else:
      print filename
      return np.array(vals)
  for i in range(0, len(vals)):
    if (np.isnan(vals[i])):
      vals[i] = 0
  min_val = min(vals)
  max_val = max(vals)
  if (max_val == min_val):
    return np.array(vals)
  else:
    return np.array([(x-min_val)/(max_val-min_val) for x in vals])

def read_interest_points(filename):
  return [int(x) for x in open(filename).readlines()]

def read_gt_interest_point(gt_interest_points_dir, basename):
  gt_interest_points = []
  shape_interest_points_filenames = glob.glob(os.path.join(gt_interest_points_dir, basename + '_*.pid'))
  for filename in shape_interest_points_filenames:
    gt_interest_points.append(read_interest_points(filename))
  return gt_interest_points

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

def compute_surface_area(V, F):
  surface_area = 0
  for i in range(len(F)):
    face = F[i]
    v0 = np.array(V[face[0]])
    v1 = np.array(V[face[1]])
    v2 = np.array(V[face[2]])
    if (len(face) >= 3):
      face_area = 0.5 * norm(np.cross(v1 - v0, v2 - v0))
      surface_area += face_area
  return surface_area

# https://github.com/gpeyre/matlab-toolboxes/tree/master/toolbox_fast_marching
def generate_geodesic_distances(filename, interest_points, output_filename, max_radius = 0.01):
  matlab_script = "\
    \nfilename = '%s'; \
    \noutput_filename = '%s'; \
    \naddpath(genpath('toolbox')); \
    \n[V,F] = read_off(filename); \
    \nnum_points = max(size(V)); \
    \ninterest_points = [%s]; \
    \nsurface_area = 0; \
    \nfor i = 1:max(size(F)); \
      \nface = F(:, i); \
      \nif (size(face, 1) >= 3); \
      \n  v1 = V(:, face(1)); \
      \n  v2 = V(:, face(2)); \
      \n  v3 = V(:, face(3)); \
      \n  face_area = 0.5 *norm(cross(v2-v1, v3-v1)); \
      \n  surface_area = surface_area + face_area; \
      \nend; \
    \nend; \
    \noptions.constraint_map = ones(num_points, 1)*sqrt(surface_area)*%d; \
    \nnum_interest_points = max(size(interest_points)); \
    \ndistances = zeros(num_interest_points, num_points); \
    \nfor j = 1:num_interest_points; \
    \n  [D,S,Q] = perform_fast_marching_mesh(V, F, interest_points(j)+1, options); \
    \n  distances(j, :) = D; \
    \nend; \
    \nsave(output_filename,'interest_points', 'distances');\
    \nquit;\
  " % (os.path.abspath(filename), os.path.abspath(output_filename), ','.join([str(x) for x in interest_points]), max_radius)
  # print matlab_script

  bin_dir = os.path.join(my_directory, 'toolbox_fast_marching')
  matlab_path = '/Applications/MATLAB_R2014b.app/bin/matlab'
  subprocess.check_call(
      'cd %s && %s -nosplash -nodisplay -r "%s"' % (bin_dir, matlab_path, matlab_script.replace('\n', ' ').replace('\r', '')),
    shell=True, stdout=open('/dev/null', 'w'))

def computeROC(basename, saliency_map, gt_interest_points, num_thresholds = 20):
  # https://en.wikipedia.org/wiki/Receiver_operating_characteristic
  # https://github.com/cvzoya/saliency/blob/master/code_forMetrics/AUC_Borji.m
  # thresholds = np.arange(0., 1., 0.05)[::-1]

  if len(gt_interest_points) == 0:
    fpr_vs_tpr = [np.zeros(num_thresholds+2)+1, np.zeros(num_thresholds+2)+1]
    cur_auc = 1
    return cur_auc, fpr_vs_tpr

  sal_at_interest_points = saliency_map[gt_interest_points]
  num_points = len(saliency_map)
  num_fixations = len(sal_at_interest_points)
  num_rand_splits = 100

  r = np.random.randint(1, num_points, size=[num_fixations, num_rand_splits]);
  randfix = saliency_map[r]; # sal map values at random locations

  # calculate AUC per random split
  fpr_vs_tpr = [np.zeros(num_thresholds+2), np.zeros(num_thresholds+2)]
  cur_auc = 0

  for s in range(num_rand_splits):
    curfix = randfix[:, s]
    if len(curfix)>0 and len(sal_at_interest_points)>0:
      thresholds = np.linspace(0, max(max(curfix), max(sal_at_interest_points)), num_thresholds)
    else:
      thresholds = np.linspace(0, 1, num_thresholds)
    fpr = np.zeros(len(thresholds)+2);
    tpr = np.zeros(len(thresholds)+2);
    fpr[0]=0; fpr[:-1] = 1; 
    tpr[0]=0; tpr[:-1] = 1; 
    for i in range (len(thresholds)):
        thresh = thresholds[i];
        fpr[i+1] = np.count_nonzero((curfix >= thresh))*1.0/num_fixations;
        tpr[i+1] = np.count_nonzero((sal_at_interest_points >= thresh))*1.0/num_fixations;
    fpr_vs_tpr[0] += fpr
    fpr_vs_tpr[1] += tpr
    cur_auc += -np.trapz(tpr, fpr)
  fpr_vs_tpr = [x/num_rand_splits for x in fpr_vs_tpr]
  cur_auc /= num_rand_splits
  return cur_auc, fpr_vs_tpr

def computeNSS(basename, saliency_map, gt_interest_points):
  if len(gt_interest_points) == 0:
    return 2.0
  stddev = np.std(saliency_map)
  if (stddev > 0):
    sal = (saliency_map - np.mean(saliency_map)) / stddev
  else:
    sal = saliency_map
  score = np.mean(sal[gt_interest_points])
  print 'NSS', score
  return score

def computeCC(basename, saliency_map, gt_saliency_map):
  if len(gt_saliency_map) == 0:
    return 1.0
  gt_sal = (gt_saliency_map - np.mean(gt_saliency_map)) / np.std(gt_saliency_map)
  stddev = np.std(saliency_map)
  if (stddev > 0):
    sal = (saliency_map - np.mean(saliency_map)) / stddev
    score = np.corrcoef(gt_sal, sal)[0][1]
  else:
    sal = saliency_map
    score = np.cov(gt_sal, sal)[0][1]
  # print 'CC', score
  return score

def computeEMD(basename, model_filename, saliency_map, gt_saliency_map, saliency_map_dir):
  # https://github.com/cvzoya/saliency/blob/master/code_forMetrics/EMD.m
  num_points = len(gt_saliency_map)
  
  # http://people.csail.mit.edu/jsolomon/  
  # "Earth Mover's Distances on Discrete Surfaces." SIGGRAPH 2014 
  bin_dir = os.path.join(my_directory, 'emdadmm')
  matlab_path = '/Applications/MATLAB_R2014b.app/bin/matlab'
  params_out_filename = os.path.join(saliency_map_dir, basename + '_emd_params.mat')
  score_out_filename = os.path.join(saliency_map_dir, basename + '_score.txt')
  savemat(params_out_filename, {'rho0': gt_saliency_map, 'rho1': saliency_map})
  matlab_script = \
  " [X, T] = readOff('%s'); \
    load '%s'; \
    rho0 = rho0'; rho1 = rho1'; \
    nEigs = 100; \
    FEM = firstOrderFEM(X,T); \
    [evecs,evals] = eigs(FEM.laplacian,FEM.vtxInnerProds,nEigs,'sm'); \
    evals = diag(evals); \
    structure = precomputeEarthMoversADMM(X, T, evecs(:,2:end)); \
    [distance,J] = earthMoversADMM(X, T, rho0, rho1, structure); \
    dlmwrite('%s', distance); quit;" % (os.path.abspath(model_filename), os.path.abspath(params_out_filename), os.path.abspath(score_out_filename))
  subprocess.check_call(
    'cd %s && %s -nosplash -nodisplay -r "%s"' % (bin_dir, matlab_path, matlab_script),
    shell=True, stdout=open('/dev/null', 'w'))
  # print 'cd %s && %s -nosplash -nodisplay -r "%s"' % (bin_dir, matlab_path, matlab_script)
  score = float(open(score_out_filename).readline().split()[0])
  os.remove(params_out_filename)
  os.remove(score_out_filename)
  return score

def computeGeodesicMap(V, F, gt_interest_points, max_dist):
  distances = geodesic.geodesic([item for v in V for item in v], [item for f in F for item in f], gt_interest_points, max_dist)
  distances = np.array(distances).reshape(len(gt_interest_points), len(V))
  distances_map = {}
  for i in range(0, len(gt_interest_points)):
    distances_map[gt_interest_points[i]] = distances[i];
  return distances_map

def computeSaliencyFromKeypoints(V, F, keypoints_per_user, geodesic_distances_map, sigma):
  saliency_map = np.zeros(len(V), dtype=np.float)
  interest = np.zeros(len(V), dtype=np.float)
  keypts = ([k for u in keypoints_per_user for k in u])
  for k in keypts:
    interest[k] += 1
  unique_keypts = list(set(keypts))

  for k in unique_keypts:
    distances = geodesic_distances_map[k]
    weight_sum = 0
    weights = []
    for i in range(len(distances)):
      if distances[i] < 1e9:
        w = np.exp(-(distances[i]*distances[i])/(2*(sigma)*(sigma)) )
        weights.append([i, w])
        weight_sum += w
    for [i,w] in weights:
      saliency_map[i] += interest[k]*w;

  min_val = min(saliency_map)
  max_val = max(saliency_map)
  return np.array([(x-min_val)/(max_val-min_val) for x in saliency_map])

def printMetricDetails(auc, label = "AUC =", confidence = 0.95):
  clean_auc = auc[~np.isnan(auc)]
  if len(clean_auc) == 0:
    auc_details = [0, 0, 0]
  else:
    auc_details = [np.mean(clean_auc), np.std(clean_auc), stats.sem(clean_auc) * stats.t._ppf((1+confidence)/2., len(clean_auc)-1)]
  print label, ' '.join([str(x) for x in auc_details])

def getModelBasenamesFromGTDir(gt_saliency_dir):
  return [os.path.basename(os.path.splitext(x)[0]) for x in sorted(glob.glob(gt_saliency_dir+'/*.val'), key=numericalSort)]

def matchSaliencyMapHistograms(saliency_map_dirs, methods, num_bins = 0, match = True):
  model_basenames = getModelBasenamesFromGTDir(saliency_map_dirs['GS'])

  # https://github.com/cvzoya/saliency/blob/master/code_forOptimization/histoMatch.m
  if (num_bins == 0):
    num_points_list = []
    for basename in model_basenames:
      gt_saliency_map = read_saliency_map(os.path.join(saliency_map_dirs['GS'], basename + '.val'))
      num_points_list.append(len(gt_saliency_map))
    num_bins = int(np.sqrt(np.median(num_points_list)))
    print 'num_bins:', num_bins, np.median(num_points_list)

  target_hist = np.zeros(num_bins)
  for basename in model_basenames:
    gt_saliency_map = read_saliency_map(os.path.join(saliency_map_dirs['GS'], basename + '.val'))
    if (len(gt_saliency_map) == 0):
      gt_saliency_map = [0 for x in range(num_bins)]
    gt_hist, gt_bin_edges = np.histogram(gt_saliency_map, bins=num_bins, density=True)
    target_hist += gt_hist
  target_hist /= len(model_basenames)
  methods_avg_hist = {m:np.zeros(num_bins) for m in methods}
  counter = 0
  for basename in model_basenames:
    # print basename
    for m in methods:
      saliency_map_dir = saliency_map_dirs[m]
      if (not os.path.isfile(os.path.join(saliency_map_dir, basename + '.val'))):
        continue
      saliency_map = read_saliency_map(os.path.join(saliency_map_dir, basename + '.val'))
      if (np.isnan(np.sum(saliency_map))):
        print 'nan: ', os.path.join(saliency_map_dir, basename + '.val')
      if (len(saliency_map) == 0):
        tmp_saliency_map = [0 for x in range(num_bins)]
        hist, bin_edges = np.histogram(tmp_saliency_map, bins=num_bins, density=True)
      else:
        hist, bin_edges = np.histogram(saliency_map, bins=len(target_hist), density=True)
      methods_avg_hist[m] += hist
      cdf = hist.cumsum()
      if (match):
        if (len(saliency_map) == 0):
          match_saliency_map  = saliency_map[:]
        else:
          new_bin_edges = np.interp(np.insert(cdf, 0, 0), np.insert(target_hist.cumsum(), 0, 0), bin_edges)
          match_saliency_map = np.interp(saliency_map, bin_edges, new_bin_edges)
        matched_map_filename = os.path.join(saliency_map_dir, basename + '_matched_hist.val')
        with open(matched_map_filename, 'w') as f:
          for val in match_saliency_map:
            f.write(str(val) + '\n')
    counter += 1
  methods_avg_hist = {k: v/counter for k,v in methods_avg_hist.iteritems()}
  methods_avg_hist['GS'] = target_hist
  return methods_avg_hist

def computeMetrics(models_dir, saliency_map_dirs, gt_interest_points_dir, metrics_per_method, roc_per_method, methods, num_thresholds, max_points = 0):
  random.seed()
  model_basenames = getModelBasenamesFromGTDir(saliency_map_dirs['GS'])
  num_models = len(model_basenames)
  if (max_points > 0):
    num_models = min(max_points, num_models)

  target_hist = np.zeros(default_num_bins)
  for basename in model_basenames:
    gt_saliency_map = read_saliency_map(os.path.join(saliency_map_dirs['GS'], basename + '.val'))
    gt_hist, gt_bin_edges = np.histogram(gt_saliency_map, bins=len(target_hist), density=True)
    target_hist += gt_hist
  target_hist /= len(model_basenames)

  for k in metrics_per_method:
    metrics_per_method[k] = {m: np.zeros(min(max_points,len(model_basenames))) for m in methods}

  # num_thresholds = 20
  for m in methods:
    roc_per_method[m]['fpr'] = np.zeros(num_thresholds+2)
    roc_per_method[m]['tpr'] = np.zeros(num_thresholds+2)
    for k in metrics_per_method:
      metrics_per_method[k][m] = np.zeros(num_models)
  
  for i in range(num_models):
    basename = model_basenames[i]
    if ((i+1)%1 == 0):
      print basename
    gt_saliency_map = read_saliency_map(os.path.join(saliency_map_dirs['GS'], basename + '.val'))
    gt_interest_points_per_user = read_gt_interest_point(gt_interest_points_dir, basename)
    gt_interest_points = list(set([x for l in gt_interest_points_per_user for x in l]))
    model_filename = os.path.join(models_dir, basename + '.off')

    for m in methods:
      eval_params = []

      if 'HS' in m:
        n1, n2 = [int(x) for x in m.split(':')[1:]]
        partitions = []
        user_ids = range(len(gt_interest_points_per_user))
        num_samples = 20
        if (n1 == 1):
          num_samples = len(user_ids)
        for l in range(num_samples):
          if (n1 == 1):
            it = [user_ids[l]]
          else:
            it = random.sample(user_ids, n1)
          rest = list(set(user_ids) - set(it))
          if (n2 <= 0):
            partitions.append({'left': it, 'right':rest})
          else:
            for ll in range(1):
              it2 = random.sample(rest, n2)
              partitions.append({'left': it, 'right':it2})
        print "num partitions: ", m, len(partitions)

        V, F = readOFF(model_filename)
        surface_area = compute_surface_area(V, F)
        sigma = 0.01*np.sqrt(surface_area)
        print len(V), surface_area
        geodesic_map = computeGeodesicMap(V, F, gt_interest_points, sigma*3)
        print 'surface:', surface_area
        for partition in partitions:
          params = {}
          left = np.array(gt_interest_points_per_user)[partition['left']]
          right = np.array(gt_interest_points_per_user)[partition['right']]

          params['saliency_map'] = computeSaliencyFromKeypoints(V, F, left, geodesic_map, sigma)
          params['gt_saliency_map'] = computeSaliencyFromKeypoints(V, F, right, geodesic_map, sigma)
          params['gt_interest_points'] = list(set([x for l in right for x in l]))

          # Match histogram
          hist, bin_edges = np.histogram(params['saliency_map'], bins=len(target_hist), density=True)
          cdf = hist.cumsum()
          new_bin_edges = np.interp(np.insert(cdf, 0, 0), np.insert(target_hist.cumsum(), 0, 0), bin_edges)
          params['saliency_map'] = np.interp(params['saliency_map'], bin_edges, new_bin_edges)

          hist, bin_edges = np.histogram(params['gt_saliency_map'], bins=len(target_hist), density=True)
          cdf = hist.cumsum()
          new_bin_edges = np.interp(np.insert(cdf, 0, 0), np.insert(target_hist.cumsum(), 0, 0), bin_edges)
          params['gt_saliency_map'] = np.interp(params['gt_saliency_map'], bin_edges, new_bin_edges)

          eval_params.append(params)
        saliency_map_dir = saliency_map_dirs['GS']
      else:
        saliency_map_dir = saliency_map_dirs[m]
        saliency_map = read_saliency_map(os.path.join(saliency_map_dirs[m], basename + '_matched_hist.val'))
        if len(saliency_map) == 0:
          saliency_map = np.asarray([0 for x in range(len(gt_saliency_map))])
          print "Empty saliency map. Set to default 0.0 saliency everywhere."
        assert(len(saliency_map) == len(gt_saliency_map))
        eval_params.append({'saliency_map': saliency_map, 'gt_saliency_map': gt_saliency_map, 'gt_interest_points': gt_interest_points})

      for k in metrics_per_method:
        metrics_per_method[k][m][i] = 0

      for params in eval_params:
        for k in metrics_per_method:
          if (k == 'auc'):
            auc, fpr_vs_tpr = computeROC(basename, params['saliency_map'], params['gt_interest_points'])
            metrics_per_method[k][m][i] += auc/len(eval_params)
            roc_per_method[m]['fpr'] += fpr_vs_tpr[0]/len(eval_params)
            roc_per_method[m]['tpr'] += fpr_vs_tpr[1]/len(eval_params)
          if (k == 'nss'):
            metrics_per_method[k][m][i] += computeNSS(basename, params['saliency_map'], params['gt_interest_points'])/len(eval_params)
          if (k == 'emd'):
            metrics_per_method[k][m][i] += computeEMD(basename, model_filename, params['saliency_map'], params['gt_saliency_map'], saliency_map_dir)/len(eval_params)
          if (k == 'cc'):
            metrics_per_method[k][m][i] += computeCC(basename, params['saliency_map'], params['gt_saliency_map'])/len(eval_params)

  for m in methods:
    roc_per_method[m]['fpr'] /= num_models
    roc_per_method[m]['tpr']  /= num_models

  for k in metrics_per_method:
    for m in methods:
      printMetricDetails(metrics_per_method[k][m], m + ' ' + k.upper() + ' =')
    print '\n'

def main():
  # Requirements:
    # Ground-truth (GS)  (Get it from Schelling Points website and uncompress it in saliency_maps/watertight_track/GS)
    # http://gfx.cs.princeton.edu/pubs/Chen_2012_SPO/SchellingData.zip

    # Cluster-based point set saliency (CS) (already present in saliency_maps/watertight_track/CS)
    
    # 3D Models: Get the 400 models from the Watertight Track Challenge (2007) and plce them in data/watertight_track/

  data_dir = 'saliency_maps/watertight_track/'
  models_dir = 'data/watertight_track/'

  gt_interest_points_dir = os.path.join(data_dir,"GS/RawData")
  saliency_map_dirs = {k: os.path.join(data_dir,'%s/Saliency'%k) for k in ['GS','RS','CS','LS','MS','PS']}
  
  max_points = 0
  num_thresholds = 20
  load_from_file = False
  match_saliency_maps = False 
  compute_metrics = (not load_from_file)
  save_metrics = (max_points == 0)
  csv_sep = ';'

  methods = ['CS']  #['RS', 'CS', 'LS', 'MS', 'PS', 'HS:1:0']
  labels = {'RS': 'Chance', 'CS': 'Tasse2015', 'LS': 'Shtrom2013', 'MS': 'Song2014', \
            'GS':'Ground-truth', 'PS': 'PCA-based', 'HS:1:0':'Human (1 vs rest)'}

  # for i in range(1, 12):
  #   m = 'HS:%d:%d' % (i,i)
  #   methods.append(m)
  #   labels[m] = "Human (%d vs %d)" % (i, i)
  
  metrics_per_method = {'auc': None, 'nss': None, 'cc': None} #,'emd': None}
  prefix = os.path.join(my_directory, "results/e_"); #"results/scans_"

  roc_per_method = {m: {'fpr': np.zeros(20), 'tpr': np.zeros(20)} for m in methods}

  if match_saliency_maps:
    methods_avg_hist = matchSaliencyMapHistograms(saliency_map_dirs, [m for m in methods if 'HS' not in m and 'GS' not in m], default_num_bins, match=False) #True)
    np.savetxt(prefix+'avg_hists.csv', np.transpose([v for k, v in methods_avg_hist.iteritems()]), \
               delimiter=csv_sep, header=csv_sep.join([k for k, v in methods_avg_hist.iteritems()]))
    print "Done matching histograms"
  
  if load_from_file:
    print 'load metrics'
    filename = prefix+'fpr_tpr.csv'
    if (os.path.isfile(filename)):
      all_roc_matrix = np.loadtxt(filename, delimiter=';', unpack=True) 
      count = 0
      for k in methods:
        for k2 in ['fpr','tpr']:
          roc_per_method[k][k2] = all_roc_matrix[count]
          count += 1
    
    for k in metrics_per_method:
      filename = prefix + k + '.csv'
      if (os.path.isfile(filename)):
        all_metric_matrix = np.loadtxt(filename, delimiter=';', unpack=True)
        count = 0
        for m in methods:
          metrics_per_method[k][m] = all_metric_matrix[count]
          count += 1

  if compute_metrics:
    computeMetrics(models_dir, saliency_map_dirs, gt_interest_points_dir, metrics_per_method, roc_per_method, methods, num_thresholds, max_points)

    if save_metrics:
      print 'save metrics'
      all_roc_matrix = [roc_per_method[k][k2] for k in methods for k2 in ['fpr','tpr']]
      all_roc_headers = [k+'_'+k2 for k in methods for k2 in ['fpr','tpr']]
      np.savetxt(prefix+'fpr_tpr.csv', np.transpose(all_roc_matrix), delimiter=csv_sep, header=csv_sep.join(all_roc_headers))

      for k2 in metrics_per_method:
        all_metric_matrix = [metrics_per_method[k2][k] for k in methods]
        all_metric_headers = [k for k in methods]
        np.savetxt(prefix+k2+'.csv', np.transpose(all_metric_matrix), delimiter=csv_sep, header=csv_sep.join(all_metric_headers))


  fig_count = 1

  if (max_points == 0):
    plt.figure(fig_count)
    fig_count += 1
    # Show cumulative distributions
    methods_avg_hist = matchSaliencyMapHistograms(saliency_map_dirs, [m for m in methods if 'HS' not in m], default_num_bins, match=False)  
    for m in methods_avg_hist:
      plt.plot(np.linspace(0, 1.0, len(methods_avg_hist[m])), (methods_avg_hist[m].cumsum()), label=labels[m], linewidth=2.0)
    plt.legend(loc='lower right')
    plt.title('Saliency cumulative distribution average' )

  plt.figure(fig_count)
  fig_count += 1
  plt.xlabel('False Positive rate')
  plt.ylabel('True Positive rate')
  plt.title ('ROC curve')
  for m in methods:
    tpr = roc_per_method[m]['tpr'] #[2:][:-1]
    fpr = roc_per_method[m]['fpr'] #[2:][:-1]
    plt.plot(fpr, tpr, marker='o', label = labels[m])
  plt.ylim([0,0.4])
  plt.xlim([0,0.4])
  plt.legend(loc='lower right')

  # for displayed_metric in metrics_per_method:
  #   plt.figure(fig_count)
  #   fig_count += 1
  #   plt.ylabel(displayed_metric.upper())
  #   plt.title (displayed_metric.upper())
  #   plt.boxplot(np.transpose([metrics_per_method[displayed_metric][k] for k in methods]))
  #   plt.xticks([1, 2, 3, 4, 5], [labels[m] for m in methods], rotation=17 )

  plt.show()

main()
