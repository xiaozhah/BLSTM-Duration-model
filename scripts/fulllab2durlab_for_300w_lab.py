# -*- coding: utf-8 -*-
home_dir = '../../../..'
import sys,os,re
sys.path.append(os.path.join(home_dir,'tools'))
sys.path.append('../model')
import numpy as np
from tqdm import tqdm
#from mxnet.gluon import nn
#from mxnet import nd,gluon,autograd,gpu,cpu
from tools import *
#from LSTM import LSTM

def Get_Ques2Regular(ques_file):
    lines = filter(lambda i:i.strip()!='', open(ques_file,'rt').readlines())
    ques_lists = []
    for line in lines:
        ques_list = []
        sub_ques = re.findall(r'[{](.*)[}]', line)[0].split(',')
        for q in sub_ques:
            q = q.replace('*',r'.*').replace('?',r'.').replace('$',r'\$')\
                 .replace('+',r'\+').replace('|',r'\|').replace('^',r'\^')
            q = re.sub(r'^([a-z])',r'^\1', q)
            # Compile pattern is Very important. Make 10X faster than originals!
            # Original(1) ques_list.append(q)
            ques_list.append(re.compile(q))
        ques_lists.append(ques_list)
    return ques_lists

def fulllab2ling(lab):
    linguistic_vec = np.zeros(len(ques_lists), dtype=np.float32)
    for i, sub_ques in enumerate(ques_lists):
        for sub_que in sub_ques:
            # Original(2) re.match(sub_que, q)
            if(sub_que.match(lab)):
                linguistic_vec[i] = 1
                break
    return linguistic_vec

def getIntDurInfo(lis):
    sum_lis=np.sum(lis)
    assert(sum_lis-np.round(sum_lis)<1e-4)
    lis_acc=np.cumsum(lis)
    result=np.zeros(5)
    result[0]=np.round(lis[0])
    if(result[0]<=0):
        result[0]=1 #保证状态时长不为零
    for i in range(1,5):
        result[i]=np.round(lis_acc[i]-np.round(np.cumsum(result[:i])[-1]))
        if(result[i]<=0):
            result[i]=1 #保证状态时长不为零
    return result

def Get_durion_readable(dur):
    state_dur_lis = np.clip(dur[:5], 1, None)
    phone_dur = np.round(dur[-1])
    if phone_dur <= 5:
        phone_dur = np.round(np.sum(state_dur_lis))
    state_dur_lis = state_dur_lis/np.sum(state_dur_lis)*phone_dur
    state_dur_lis = state_dur_lis.astype(np.float64)
    
    state_dur_lis = getIntDurInfo(state_dur_lis)
    dur_lis = np.append(state_dur_lis, np.sum(state_dur_lis)).astype(np.int)
    return dur_lis

def Gen_durlab_From_durtxt(fullab_dir,durtxt_dir,durlab_dir):
    # fullab_dir and durtxt_dir is Input
    # durlab_dir is Output
    SaveMkdir(durlab_dir)
    for basename in tqdm(sorted(os.listdir(fullab_dir))):
        name = basename.split('.')[0]
        tqdm.write('process %s' % name)
        fulllab = open(os.path.join(fullab_dir, name+'.lab'),'rt').readlines()
        cutedlab_file = os.path.join(durlab_dir, name+'.dur')
        durtxt_file = open(os.path.join(durtxt_dir, name+'.txt'),'rt')
        durtxt_file.readline()
        with open(cutedlab_file,'wt') as fp_cutedlab:
            for line in fulllab:
                lis = durtxt_file.readline()
                if lis == '':
                    break
                else:
                    duration = np.array(lis.split('\t'),dtype=np.int)
                    assert(np.sum(duration[:5])==duration[-1])
                    for i, dur in enumerate(duration[:-1]):
                        fp_cutedlab.write('%s    %d    %d\n' % (line.rstrip(),i+2,dur))
        durtxt_file.close()

if __name__ == '__main__':
    fulllab2durfile = 0
    durfile2durlab = 1

    if fulllab2durfile:
        labdir   = '../../analyse_corpus/data/300w_lab/fulllab'
        que_file = './questions.hed'
        test_dur_lab = '../data/test_dur_300w_lab'
        #分成六份并行生成
        ref_lst = '../../analyse_corpus/data/lst_6_part/300w_lab_3.lst'
        SaveMkdir(test_dur_lab)

        net = LSTM()
        ctx = gpu()
        net.load_params('../model/LSTM.parms',ctx=ctx)
        mean_out_file = '../data/MeanStd_BLSTM_label.mean'
        mean_out = ReadFloatRawMat(mean_out_file, 6)

        ques_lists = Get_Ques2Regular(que_file)
        ref_files = np.loadtxt(ref_lst,'str')
        for basename in tqdm(ref_files):
            tqdm.write('process %s' % basename)
            lab_file = os.path.join(labdir, basename+'.lab')
            test_dur_file = os.path.join(test_dur_lab, basename+'.txt')
            labs = open(lab_file,'rt').readlines()
            linguistic_Mat = nd.array(np.r_[map(fulllab2ling,labs)], ctx=ctx)
            outData = net(linguistic_Mat.expand_dims(axis=1)).flatten().asnumpy()
            outData = DeNormalization_MeanStd(outData,range(6),mean_out)
            with open(test_dur_file, 'wt') as test_dur_fp:
                test_dur_fp.write('S1\tS2\tS3\tS4\tS5\tPhone\n')
                for i in outData:
                    test_dur_fp.write('%d\t%d\t%d\t%d\t%d\t%d\n' % tuple(Get_durion_readable(i)))

    if durfile2durlab:
        test_fulllab_dir = '../../analyse_corpus/data/300w_lab/fulllab'
        test_durtxt_dir = '../data/test_dur_300w_lab'
        Gen_test_durlab_dir = '../../analyse_corpus/data/300w_lab/durlab_BLSTM'
        Gen_durlab_From_durtxt(test_fulllab_dir, test_durtxt_dir, Gen_test_durlab_dir)
