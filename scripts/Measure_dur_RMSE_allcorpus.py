# -*- coding: utf-8 -*-
import numpy as np

ref_dur_file = '../data/train+val_duration.txt'
pred_dur_file = '../data/train+val predicted duration.txt'
phone_name_frame_id = '../data/phone_name_frame_id.csv'

ref = np.loadtxt(ref_dur_file,dtype=float,skiprows=1)
predict = np.loadtxt(pred_dur_file,dtype=float,skiprows=1)

print "全部音素"
print "Duration RMSE:\nS1:%f frame\nS2:%f frame\nS3:%f frame\nS4:%f frame\nS5:%f frame\nPhone:%f frame\n" % tuple(np.sqrt(np.mean((ref-predict)**2,axis=0)))

phone = np.loadtxt(phone_name_frame_id,dtype='str')[:,1]
index = np.where(~((phone == 'sil') | (phone == 'sp')))
print "非静音(sil sp)音素"
print "Duration RMSE:\nS1:%f frame\nS2:%f frame\nS3:%f frame\nS4:%f frame\nS5:%f frame\nPhone:%f frame\n" % tuple(np.sqrt(np.mean((ref[index]-predict[index])**2,axis=0)))
