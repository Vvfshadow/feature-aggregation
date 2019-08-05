import torch
from tqdm import tqdm
import os
from scipy.spatial.distance import cdist
import json
from data_pose import Data as Data_pose
from network_pose import MGN as MGN_pose
from utils.extract_feature_pose import extract_feature as extract_feature_pose
import numpy as np
from data import Data
from network import MGN
from utils.extract_feature import extract_feature
from utils.metrics import re_ranking

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def result(distmat, query_ids=None, gallery_ids=None,
           query_cams=None, gallery_cams=None, title=None):
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)

    dd = []
    for i in range(m):
        # Filter out the same id and same camera
        d = {}
        d['query_id'] = query_ids[i].astype(np.int32).tolist()
        valid = ((gallery_ids[indices[i]] != query_ids[i]) &
                 (gallery_cams[indices[i]] != query_cams[i]))
        ans_ids = gallery_ids[indices[i]][valid]
        d['ans_ids'] = ans_ids.astype(np.int32).tolist()
        dd.append(d)
    with open(epoch_json + '/' + title + '.json', 'w', encoding='utf-8') as json_file:
        json.dump(dd, json_file, ensure_ascii=False)
    print('json finished')



pose_path = 'weights_pose/model.pt'
jitter_path = 'weights/model.pt'

data_pose = Data_pose()
data_jitter = Data()

model_pose = MGN_pose().to('cuda')
model_jitter = MGN().to('cuda')

print('start evaluate')
model_pose.load_state_dict(torch.load(pose_path))
model_jitter.load_state_dict(torch.load(jitter_path))

model_pose.eval()
model_jitter.eval()

qf_pose = extract_feature_pose(model_pose, tqdm(data_pose.query_loader)).numpy()
gf_pose = extract_feature_pose(model_pose, tqdm(data_pose.test_loader)).numpy()

qf_jitter = extract_feature(model_jitter, tqdm(data_jitter.query_loader)).numpy()
gf_jitter = extract_feature(model_jitter, tqdm(data_jitter.test_loader)).numpy()


qf = np.concatenate((qf_pose, qf_jitter), 1)
gf = np.concatenate((gf_pose, gf_jitter), 1)

qf = np.squeeze(np.array(qf))
# feature norm
q_n = np.linalg.norm(qf, axis=1, keepdims=True)
qf = qf / q_n

gf = np.squeeze(np.array(gf))
# feature norm
g_n = np.linalg.norm(gf, axis=1, keepdims=True)
gf = gf / g_n

epoch_json = 'metric_final'
os.makedirs(epoch_json)

#########################no re rank##########################
dist = cdist(qf, gf)
result(dist, data_jitter.queryset.ids, data_jitter.testset.ids, title='without rerank')

#########################   re rank##########################
q_g_dist = np.dot(qf, np.transpose(gf))
q_q_dist = np.dot(qf, np.transpose(qf))
g_g_dist = np.dot(gf, np.transpose(gf))
dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
result(dist, data_jitter.queryset.ids, data_jitter.testset.ids, title='rerank')

#########################   query expansion##########################
qf_new = []
T = 9
for i in range(len(dist)):
    indice = np.argsort(dist[i])[:T]
    temp = np.concatenate((qf[i][np.newaxis, :], gf[indice]), axis=0)
    qf_new.append(np.mean(temp, axis=0, keepdims=True))

qf = np.squeeze(np.array(qf_new))
# feature norm
q_n = np.linalg.norm(qf, axis=1, keepdims=True)
qf = qf / q_n

q_g_dist = np.dot(qf, np.transpose(gf))
q_q_dist = np.dot(qf, np.transpose(qf))
g_g_dist = np.dot(gf, np.transpose(gf))
dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
# np.save('metric_label/dist.npy', dist)
# np.save('metric_label/query_ids.npy', data_jitter.queryset.ids)
# np.save('metric_label/gallery_ids.npy', data_jitter.testset.ids)

result(dist, data_jitter.queryset.ids, data_jitter.testset.ids, title='query_expansion')

