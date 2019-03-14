# -*- coding: utf-8 -*-
home_dir = '../../../..'
import sys,os
sys.path.append(os.path.join(home_dir,'tools'))
from tools import *
from functions import *
import numpy as np
import os,struct
from glob import glob

def GetDurFile(cutedlabDir, dur_dir, dur_file):
    SaveMkdir(dur_dir)
    with open(dur_file,'wt') as dur_fp:
        dur_fp.write('S1\tS2\tS3\tS4\tS5\tPhone\n')
        for item in sorted(os.listdir(cutedlabDir)):
            print item
            cutedLabFile = cutedlabDir+os.sep+item
            dur_dir_file = open(os.path.join(dur_dir, item[:-4]+'.dat'),'wb')
            lines = open(cutedLabFile,'rt').read().splitlines() 
            for i in range(len(lines)):
                if lines[i]=='':continue
                if lines[i].split()[2].endswith('[2]'):
                    phone_start = int(round(float(lines[i].split()[0])/50000.0))
                    phone_end = int(round(float(lines[i+4].split()[1])/50000.0))
                    state_dur_lis = [int(round(float(lines[j].split()[1])/50000.0))-int(round(float(lines[j].split()[0])/50000.0)) for j in range(i,i+5)]
                    dur = np.append(state_dur_lis,phone_end-phone_start).tolist()
                    dur_dir_file.write(struct.pack('<6f',*dur))
                    dur_fp.write('%d\t%d\t%d\t%d\t%d\t%d\n' % tuple(dur))
            dur_dir_file.close()

if __name__=='__main__':
    GetLinguistic = 0 # Run ../data/context_bin_out.pl
    GetDur = 0
    norm = 0
    Gen_cutedlab = 0
    
    if GetDur:
        cutedlabDir = r'F:\xzhou\Yanping13k_IFLYFE\labels\cutedlab'
        dur_file = '../data/train+val_duration.txt' #包括五个状态和一个音素时长
        dur_dir = '../data/train+val_duration'
        GetDurFile(cutedlabDir,dur_dir,dur_file)

    if norm:
        DNNoutDir   = '../data/train+val_duration'
        ############## normalization ##############
        #同时输出均值文件
        Normalization_MeanStd_Dir(DNNoutDir,  6, range(6), None, 'BLSTM', 'label', ref_file = os.path.join(home_dir,'train_file.lst'))
        mean_file_label = ReadFloatRawMat('../data/MeanStd_BLSTM_label.mean',6)
        #归一化验证集
        Normalization_MeanStd_Dir(DNNoutDir,  6, range(6), mean_file_label,'BLSTM','label',ref_file = os.path.join(home_dir,'val_file.lst'))

    #Train model then goto test part
