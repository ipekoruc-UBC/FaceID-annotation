#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 01:17:24 2019

@author: mkhademi
"""
import sys

import numpy as np
import statistics
import scipy.io as sio
sys.setrecursionlimit(50000)
import pickle
import os
import math
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

external1='/Users/ipekoruc/ipekoruc/ipek/Mahmoud/FaceDiet'

results = external1 + '/Face_Diet_Oruc/output/results/'
ann_folder = external1 + '/Face_Diet_Oruc/annotations/'
output_graphs = external1 + '/Face_Diet_Oruc/output/graphs/'
output_align = external1 + '/Face_Diet_Oruc/output/align/'
img_folder = external1 + '/Face_Diet_Oruc/'
img_folder = external1 + '/Face_Diet_Oruc/'
all_subjs = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
             'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20',
             'S21', 'S22', 'S23', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31']

subjs_gender = ['f', 'm', 'f', 'f', 'm', 'm', 'f', 'm', 'f', 'f', 'm', 'f',
                'm', 'm', 'f', 'm', 'm', 'f', 'f', 'f', 'm', 'f', 'm', 'm',
                'm', 'f', 'm', 'm', 'm', 'f']

subjs_ethnic = ['cau', 'afr', 'cau', 'cau', 'cau', 'cau', 'cau', 'cau', 'cau',
               'easia', 'cau', 'cau', 'cau', 'cau', 'cau', 'cau', 'cau', 'cau',
               'cau', 'cau', 'cau', 'cau', 'sasia', 'cau', 'cau', 'cau', 'cau',
               'cau', 'cau', 'cau']
sys.stdout = open(results + 'results_stranger.txt', 'w')
a_b_dict = {('f', 'fr'): (8.4, -0.86), ('m', 'fr'): (8.3, -0.83), ('f', '3/4'): (8.2, -0.86),
            ('m', '3/4'): (8.1, -0.84), ('f', 'pr'): (8.7, -0.86), ('m', 'pr'): (8.0, -0.72)}

all_number_of_unique_ids = []
all_number_of_faces = []
all_number_of_hours_of_footage = []
all_fam_strs = []
all_m_f = []
all_str_perc = []
all_top1 = []
all_top2 = []
all_top3 = []
all_top4 = []
all_top5 = []
all_top6 = []
all_top123 = []
all_gender_top1 = []
all_gender_top2 = []
all_gender_top3 = []
all_positions_top1 = []
all_positions_top2 = []
all_positions_top3 = []
all_ethnics_top1 = []
all_ethnics_top2 = []
all_ethnics_top3 = []
all_inconsistent = []
all_dist_top1 = []
all_dist_top2 = []
all_dist_top3 = []
all_avg_dist_same_ethnic = []
all_avg_dist_other_ethnic = []
all_inconsistent_faces = []
subj_index = 0
global_dist_top1 = []
global_dist_top2 = []
global_dist_top3 = []
global_dist_same_ethnic = []
global_dist_other_ethnic = []
all_afr_count = []
all_cau_count = []
all_easia_count = []
all_Call_count = []
for subj in all_subjs:
    ann_index = 8
    gender_ann_index = 6
    position_ann_index = 7
    ethnic_ann_index = 9
    bbx_ann_index = 2
    if subj == 'S02':
        ann_index = 6
        gender_ann_index = 4
        position_ann_index = 5
        ethnic_ann_index = 7
    with open(output_graphs + subj + '.pickle', 'rb') as handle:
        output_list = pickle.load(handle)
    print(subj, ':')
    total_images = 0.0
    for k in os.listdir(img_folder + subj +' (Oruc, Ipek)/'):
        if k.endswith('.jpg'):
           total_images += 1
    total_faces = 0.0
    for k in os.listdir(output_align + subj + '/'):
        if k.endswith('.png'):
           total_faces += 1
    all_number_of_faces.append(total_faces)
    anns = sio.loadmat(ann_folder + subj + '_PL.mat')
     
    str1 = '# of unique ids: ' + str(len(output_list))
    str2 = '# of unique ids per face: ' + str("{0:.3f}".format(len(output_list)/total_faces))
    str3 = '# of unique ids per image: ' + str("{0:.3f}".format(len(output_list)/total_images))
    durations = []
    fam_str = []
    m_f = []
    positions = []
    ethnics = []
    inconsistent = []
    num_inconsistent = []
    dists = []

    for identity in output_list:
        num_fam = 0.0 
        num_str = 0.0 
        for face in identity:
            for i in range(len(anns['facedetec1'][0])):
                if anns['facedetec1'][0][i][1][0] == face[:8] + '.jpg':                   
                    if len(anns['facedetec1'][0][i][ann_index]) > 0 and len(anns['facedetec1'][0][i][ann_index][0][0]) > 0:
                        if anns['facedetec1'][0][i][ann_index][int(face[9:-6])][0][0] == 'fam':
                            num_fam += 1
                        else:
                            num_str += 1 
                    break
   
        fam_str.append(round(num_fam/(num_str+num_fam), 2))
        if round(num_fam/(num_str+num_fam), 2) > 0 and round(num_fam/(num_str+num_fam), 2) < 1:
            inconsistent.append((str(int(num_fam)) + '/' + str(int(num_str))))
        temp = min(num_str, int(num_fam))
        num_inconsistent.append(temp)
        durations.append((num_str+num_fam-temp) * 30)        
    inconsistent_faces = []
    all_inconsistent.append(inconsistent)
    all_fam_strs.append(fam_str)
    j = 0
    for identity in output_list:
        for face in identity:           
            for i in range(len(anns['facedetec1'][0])):
                if anns['facedetec1'][0][i][1][0] == face[:8] + '.jpg':                   
                    if len(anns['facedetec1'][0][i][ann_index]) > 0 and len(anns['facedetec1'][0][i][ann_index][0][0]) > 0:
                        if anns['facedetec1'][0][i][ann_index][int(face[9:-6])][0][0] == 'fam':
                            if fam_str[j] < 0.5:
                                inconsistent_faces.append(face)
                        else:
                            if fam_str[j] >= 0.5:
                                inconsistent_faces.append(face) 
                    else:
                        inconsistent_faces.append(face)
                                
                    break
        j += 1  
    all_inconsistent_faces.append(inconsistent_faces[:])
 
    frames_id = []
    temp_dists = []
    for identity in output_list:
        num_male = 0.0
        position_dict = {'fr': 0.0, 'pr': 0.0, '3/4': 0.0, 'f': 0.0, 'Call': 0.0}
        position_list = []
        ethnic_dict = {'afr': 0.0, 'cau': 0.0, 'easia': 0.0, 'e': 0.0, 'c': 0.0, 'Call': 0.0, 'ea': 0.0}
        dist = 0.0
        dist_count = 0
        id_dists = []
        frames = []
        for face in identity:
            if face in inconsistent_faces:
                continue
            exp_log_d = 0.0
            for i in range(len(anns['facedetec1'][0])):
                if anns['facedetec1'][0][i][1][0] == face[:8] + '.jpg': 
                    frames.append(face[:8] + '.jpg')
                    flag_male = 'f'
                    if anns['facedetec1'][0][i][gender_ann_index][int(face[9:-6])][0][0] == 'm':
                        num_male += 1
                        flag_male = 'm'
                    cur_pos = anns['facedetec1'][0][i][position_ann_index][int(face[9:-6])][0][0]
                    position_dict[cur_pos] += 1
                    position_list.append(cur_pos)
                    ethnic_dict[anns['facedetec1'][0][i][ethnic_ann_index][int(face[9:-6])][0][0]] += 1
                    bbx = anns['facedetec1'][0][i][bbx_ann_index][int(face[9:-6])]
                    #print(bbx)
                    if i < len(anns['segments'][0]):
                        #print(anns['segments'][0][i][0])
                        for l in range(len(anns['segments'][0][i][0])):
                            if len(anns['segments'][0][i][0][0]) > 0:
                                x1 = anns['segments'][0][i][0][l][0]
                                y1 = anns['segments'][0][i][0][l][1]
                                x2 = anns['segments'][0][i][0][l][2]
                                y2 = anns['segments'][0][i][0][l][3]
                                if (x1 >= bbx[0] and y1 >=bbx[1] and x2 <= bbx[0]+bbx[2] and y2 <=bbx[1]+bbx[3]) or \
                                    (x2 >= bbx[0] and y2 >=bbx[1] and x1 <= bbx[0]+bbx[2] and y1 <=bbx[1]+bbx[3]):      
                                    log_l = math.log(math.sqrt((x1-x2)**2 + (y1-y2)**2))
                                    if cur_pos in ['fr', 'pr', '3/4']:
                                        temp_a, temp_b = a_b_dict[(flag_male, cur_pos)]
                                        exp_log_d = math.exp((log_l - temp_a) / temp_b)
                                        id_dists.append(exp_log_d)
                                        dist_count += 1
                                    break
                    break
            dist += exp_log_d
        m_f.append(round(num_male/len(identity), 2))
        if dist_count == 0:
            dists.append(float('nan'))
        else:
            dists.append(dist/dist_count)
            #print(dist/dist_count)
        #print(position_dict)
        temp_dists.append(id_dists)
        #positions.append(max(position_dict, key=position_dict.get))
        sum_position = position_dict['fr'] + position_dict['pr'] + position_dict['3/4']
        if sum_position > 0.0:
            positions.append((100*position_dict['fr']/sum_position, 100*position_dict['pr']/sum_position, 100*position_dict['3/4']/sum_position))
            #positions.append((position_dict['fr'],position_dict['pr'],position_dict['3/4']))
        else:
            positions.append((0,0,0))
        #print(ethnic_dict)
        frames_id.append(frames)
        ethnics.append(max(ethnic_dict, key=ethnic_dict.get))
    
    print('Number of images', total_images)
    print('Number of faces:', total_faces)
    print('Number of unique identities:', len(durations))
    print('Duration of each identity:', sorted(durations, reverse=True))
    print('Avg duration: ' + str("{0:.3f}".format(statistics.mean(durations))))
    print('Std duration: ' + str("{0:.3f}".format(np.sqrt(np.var(durations)))))
    all_number_of_unique_ids.append(len(durations))
    all_number_of_hours_of_footage.append(int((total_images*30.0) // 3600))
    objects = []
    for i in range(len(durations)):
        objects.append('I' + str(i+1))
    y_pos = np.arange(len(objects))
     
    plt.bar(y_pos, sorted(durations, reverse=True), align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.xlabel('Identity')
    plt.ylabel('Duration of exposure (second)')
    plt.title('Duration of exposure for each identity (' + subj + ')\n' + str1 + ', ' + str2 + ',\n' + str3)
    plt.savefig(results + subj + '.png')  
    plt.show() 
    
    exposure_time = sum(durations)
    sum_time = 0.0
    identity = 0
    while sum_time < exposure_time * 0.90:
        sum_time += sorted(durations, reverse=True)[identity]
        identity += 1
    print('# of unique individuals accounting for 90% of exposure time-wise (' + subj + '): ', identity)
    indices = [i[0] for i in sorted(enumerate(durations), key=lambda x:x[1], reverse=True)]
    fam_count = 0
    for i in range(identity):
        if fam_str[indices[i]] >= 0.5:
            fam_count += 1
    
    print('# of unique familiar individuals accounting for 90% of exposure time-wise (' + subj + '): ', fam_count)
    fam_str2 = [fam_str[i] for i in indices]
    m_f2 = [m_f[i] for i in indices]
    positions2 = [positions[i] for i in indices]
    dists2 = [dists[i] for i in indices]
    ethnics2 = [ethnics[i] for i in indices]
    num_inconsistent2 = [num_inconsistent[i] for i in indices]
    temp_dists2 = [temp_dists[i][:] for i in indices]
    frames_id2 = [frames_id[i][:] for i in indices]
    
    ethnics_fam_count = {'afr': 0.0, 'cau': 0.0, 'easia': 0.0, 'e': 0.0, 'c': 0.0, 'Call': 0.0, 'ea': 0.0}
    for i in range(len(ethnics2)):
        if fam_str2[i] < 0.5:
            ethnics_fam_count[ethnics2[i]] += 1
            if ethnics2[i] == 'Call':
                print(frames_id2[i])
    print(ethnics_fam_count)
    all_afr_count.append(ethnics_fam_count['afr'])
    all_cau_count.append(ethnics_fam_count['cau'])
    all_easia_count.append(ethnics_fam_count['easia'])
    all_Call_count.append(ethnics_fam_count['Call'])
    
    print('Familiar or Stranger (for each identity)?:', fam_str2)
    all_m_f.append(m_f2)
    str_tot, tot = 0.0, 0.0
    durations_sorted = sorted(durations, reverse=True)
    for i in range(len(durations)):
        tot += durations_sorted[i]
        str_tot += durations_sorted[i] * float(fam_str2[i] < 0.5)
    print('total exposure duration (%) accounted for by familiar faces:', round(100*str_tot/tot,1))
    all_str_perc.append(round(100*str_tot/tot, 1))
    top = 6
    j = 0
    top_list = []
    for i in range(len(durations)):
        if fam_str2[i] < 0.5 and j < top:
            j += 1
            top_list.append(i)
    print('info:')
    i = 0
    for identity in output_list:
        rank = '-' 
        if len(top_list) > 0 and i == indices[top_list[0]]:
            rank = 'top'
        if len(top_list) > 1 and i == indices[top_list[1]]:
            rank = 'top 2nd'
        if len(top_list) > 2 and i == indices[top_list[2]]:
            rank = 'top 3rd'          
        if fam_str[i] > 0.5:
            temp = 'fam'
        else:
            temp = 'str'
        if m_f[i] > 0.5:
            temp2 = 'm'
        else:
            temp2 = 'f'
        print('ID' + str(i) + ':' + temp + ',' + ethnics[i] + ',' + temp2 + ',' +
              str(int(len(identity)-num_inconsistent[i])) + ',' + rank)
        i += 1    
    
    
    if len(top_list) > 0:
        print('total exposure duration (%) accounted for by the top familiar face:', round(100*durations_sorted[top_list[0]]/tot, 1))
        all_top1.append(round(100*durations_sorted[top_list[0]]/tot, 1))
        if m_f2[top_list[0]] >= 0.5:
            all_gender_top1.append('m')
        else:
            all_gender_top1.append('f')
        all_positions_top1.append(positions2[top_list[0]])
        all_ethnics_top1.append(ethnics2[top_list[0]])
        all_dist_top1.append(dists2[top_list[0]])
        global_dist_top1.extend(temp_dists2[top_list[0]])
        top123=100*(durations_sorted[top_list[0]]) /tot
    else:
        all_top1.append(0.0)
        all_gender_top1.append('x')
        all_positions_top1.append((0,0,0))
        all_ethnics_top1.append('x')
        all_dist_top1.append(0.0) 
        top123 = 0
    if len(top_list) > 1:
        all_top2.append(round(100*durations_sorted[top_list[1]]/tot, 1))
        if m_f2[top_list[1]] >= 0.5:
            all_gender_top2.append('m')
        else:
            all_gender_top2.append('f')
        all_positions_top2.append(positions2[top_list[1]])
        all_ethnics_top2.append(ethnics2[top_list[1]])
        all_dist_top2.append(dists2[top_list[1]]) 
        global_dist_top2.extend(temp_dists2[top_list[1]])
        top123=100*(durations_sorted[top_list[0]] + durations_sorted[top_list[1]]) /tot
    else:
        all_top2.append(0.0)
        all_gender_top2.append('x')
        all_positions_top2.append((0,0,0))
        all_ethnics_top2.append('x')
        all_dist_top2.append(0.0) 
            
    if len(top_list) >= 3:
        global_dist_top3.extend(temp_dists2[top_list[2]])
        all_top3.append(round(100*durations_sorted[top_list[2]]/tot, 1))
        all_dist_top3.append(dists2[top_list[2]]) 
        if m_f2[top_list[2]] >= 0.5:
            all_gender_top3.append('m')
        else:
            all_gender_top3.append('f')
        top123=100*(durations_sorted[top_list[0]] + durations_sorted[top_list[1]] + durations_sorted[top_list[2]]) /tot
        all_positions_top3.append(positions2[top_list[2]])
        all_ethnics_top3.append(ethnics2[top_list[2]])
    else:
        all_top3.append(0.0)
        all_gender_top3.append('x')
        all_positions_top3.append((0,0,0))
        all_ethnics_top3.append('x')
        all_dist_top3.append(0.0) 
    if len(top_list) >= 4:
        all_top4.append(round(100*durations_sorted[top_list[3]]/tot, 1))
    else:
        all_top4.append(0.0)
    if len(top_list) >= 5:
        all_top5.append(round(100*durations_sorted[top_list[4]]/tot, 1))
    else:
        all_top5.append(0.0)
    if len(top_list) >= 6:
        all_top6.append(round(100*durations_sorted[top_list[5]]/tot, 1))
    else:
        all_top6.append(0.0)
            
    print('total exposure duration (%) accounted for by the top familiar face:', top123)
    all_top123.append(round(top123, 1))
    sum_dist_same_ethnic = 0.0
    count_dist_same_ethnic = 0.0
    sum_dist_other_ethnic = 0.0
    count_dist_other_ethnic = 0.0
    for i in range(len(dists2)):
        if ethnics2[i] in ['afr','cau', 'easia']:
            if dists2[i] is not None and fam_str2[i] < 0.5 and ethnics2[i] == subjs_ethnic[subj_index]:
                sum_dist_same_ethnic += dists2[i]
                count_dist_same_ethnic += 1
                global_dist_same_ethnic.extend(temp_dists2[i][:])
            if dists2[i] is not None and fam_str2[i] < 0.5 and ethnics2[i] != subjs_ethnic[subj_index]:
                sum_dist_other_ethnic += dists2[i]
                count_dist_other_ethnic += 1
                global_dist_other_ethnic.extend(temp_dists2[i])
    if count_dist_same_ethnic > 0:
        all_avg_dist_same_ethnic.append(sum_dist_same_ethnic/count_dist_same_ethnic)
    else:
        all_avg_dist_same_ethnic.append(0.0)
    if count_dist_other_ethnic:
        all_avg_dist_other_ethnic.append(sum_dist_other_ethnic/count_dist_other_ethnic)
    else:
        all_avg_dist_other_ethnic.append(0.0)
    subj_index += 1         
    print('********************************************************************')
print('########################################################################') 
print('Number of unique identities after each hour:')          
shape_color = ['-xb', '-xg', '-xr', '-xc', '-xm', '-xy', '-xk', '-+r', '-^b', '-^g',
               '-^r', '-^c', '-^m', '-^y', '-^k', '-+g','-sb', '-sg', '-sr', '-sc',
               '-sm', '-sy', '-sk', '-+b', '-ob', '-og', '-or', '-oc', '-om', '-oy'] 
z = 0              
for subj in all_subjs:
    with open(output_graphs + subj + '.pickle', 'rb') as handle:
        output_list = pickle.load(handle)
    anns = sio.loadmat(ann_folder + subj + '_PL.mat')
    t = 0.0
    y = []
    y_fam = []
    x = []
    saw = [0] * len(output_list)
    num_ids_saw = 0
    num_ids_saw_fam = 0
    for i in range(len(anns['facedetec1'][0])):
        t += 30.0 / 3600.0
        x.append(t)
        num_new = 0
        num_new_fam = 0
        if len(anns['facedetec1'][0][i][2]) > 0:
            for j in range(len(anns['facedetec1'][0][i][2])):
                face = anns['facedetec1'][0][i][1][0][:-4] + '_' + str(j) + '_0.png'
                for k in range(len(output_list)):
                    if face in output_list[k]:
                        if saw[k] == 0:
                            saw[k] = 1
                            num_new += 1
                            if all_fam_strs[z][k] >= 0.5:
                                num_new_fam += 1
                        break
        num_ids_saw += num_new
        num_ids_saw_fam += num_new_fam
        y.append(num_ids_saw)
        y_fam.append(num_ids_saw_fam)
    y2 = [y[k] for k in range(0, len(anns['facedetec1'][0]), 120)]
    y2_fam = [y_fam[k] for k in range(0, len(anns['facedetec1'][0]), 120)]

    plt.ylabel('Number of unique familiar identities')  
    plt.xlabel('Time (hour)')
    plt.plot(y2_fam, shape_color[z], label=subj) 
    z += 1 
    print(subj + ': ', y2)          
plt.legend(frameon=True, ncol=3, loc='upper right', bbox_to_anchor=(1, 0.93), fontsize=7.5)              
plt.savefig(results + 'results.png')                 
plt.show()               
                              
x = all_number_of_hours_of_footage
y = all_number_of_unique_ids
area = 40
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'r', 'b', 'g',
               'r', 'c', 'm', 'y', 'k', 'g','b', 'g', 'r', 'c',
               'm', 'y', 'k', 'b', 'b', 'g', 'r', 'c', 'm', 'y'] 

markers = ['x', 'x', 'x', 'x', 'x', 'x', 'x', '+', '^', '^',
               '^', '^', '^', '^', '^', '+','s', 's', 's', 's',
               's', 's', 's', '+', 'o', 'o', 'o', 'o', 'o', 'o'] 
for i in range(len(all_number_of_hours_of_footage)):
    plt.scatter(x[i], y[i], c=colors[i], s=area, marker=markers[i], label=all_subjs[i], alpha=0.5)
plt.title('Scatter plot')
plt.xlabel('Number of hours of footage')
plt.ylabel('Number of unique ids')
plt.legend(frameon=True, ncol=3, loc='upper right', bbox_to_anchor=(1, 0.93), fontsize=7.5) 
plt.savefig(results + 'scatter.png')  
plt.show()               
#print('Number of hours of footage for all subjects:', all_number_of_hours_of_footage) 
print('Number of unique identities for all subjects:', all_number_of_unique_ids)                 
                              
all_number_of_unique_ids_fam = [] 
all_number_of_unique_ids_strg = []               
for z in all_fam_strs:
    count = 0
    strg = 0
    for v in z:
        if v >= 0.5:
            count += 1
        else:
            strg += 1
    all_number_of_unique_ids_fam.append(count)
    all_number_of_unique_ids_strg.append(strg)
                    
y = all_number_of_unique_ids_fam
for i in range(len(all_number_of_hours_of_footage)):
    plt.scatter(x[i], y[i], c=colors[i], s=area, marker=markers[i], label=all_subjs[i], alpha=0.5)
plt.title('Scatter plot (fimiliar)')
plt.xlabel('Number of hours of footage')
plt.ylabel('Number of unique familiar ids')

plt.legend(frameon=True, ncol=3, loc='upper right', bbox_to_anchor=(1, 0.93), fontsize=7.5) 
plt.savefig(results + 'scatter_familiar.png')  
plt.show()               
#print('Number of hours of footage for all subjects:', all_number_of_hours_of_footage) 
print('Number of unique familiar identities for all subjects:', all_number_of_unique_ids_fam)  
print('Number of unique stranger identities for all subjects:', all_number_of_unique_ids_strg) 
print('all_stranger_percentage:', all_str_perc)
i = 0
print('inconsistents:')
for x in all_inconsistent:
    print(all_subjs[i]+':', x)
    i += 1

print('Exposure duration accounted for by the TOP1 stranger face id:')
for x in all_top1:
    print(x)
print('Exposure duration accounted for by the TOP2 stranger face id:')
for x in all_top2:
    print(x)
print('Exposure duration accounted for by the TOP3 stranger face id:')
for x in all_top3:
    print(x)
print('Exposure duration accounted for by the TOP4 stranger face id:')
for x in all_top4:
    print(x)
print('Exposure duration accounted for by the TOP5 stranger face id:')
for x in all_top5:
    print(x)
print('Exposure duration accounted for by the TOP6 stranger face id:')
for x in all_top6:
    print(x)
print('Exposure duration accounted for by the TOP123 stranger face id:')
for x in all_top123:
    print(x)
print('Gender of the TOP1 stranger face id:')     
for x in all_gender_top1:
    print(x)
print('TOP1 stranger id same or opposite gender of the participant:')
for x, y in zip(all_gender_top1, subjs_gender):
    if x == y:
        print('s')
    else:
        print('o')
print('Gender of the TOP2 stranger face id:')     
for x in all_gender_top2:
    print(x)
print('TOP2 stranger id same or opposite gender of the participant:')
for x, y in zip(all_gender_top2, subjs_gender):
    if x == y:
        print('s')
    else:
        print('o')
print('Gender of the TOP3 stranger face id:')    
for x in all_gender_top3:
    print(x)
print('TOP3 stranger id same or opposite gender of the participant:')
for x, y in zip(all_gender_top3, subjs_gender):
    if x == y:
        print('s')
    else:
        print('o')  

print('Pose of the TOP1 stranger face id (percentage of fr, pr, 3/4):')     
for x in all_positions_top1:
    print((round(x[0], 1), round(x[1], 1), round(x[2], 1)))

print('Pose of the TOP2 stranger face id (percentage of fr, pr, 3/4):')      
for x in all_positions_top2:
    print((round(x[0], 1), round(x[1], 1), round(x[2], 1)))

print('Pose of the TOP3 stranger face id (percentage of fr, pr, 3/4):')    
for x in all_positions_top3:
    print((round(x[0], 1), round(x[1], 1), round(x[2], 1)))


print('Ethnicity of the TOP1 stranger face id:')     
for x in all_ethnics_top1:
    print(x)
print('TOP1 stranger id same or other race/ethnicity of the participant:')
for x, y in zip(all_ethnics_top1, subjs_ethnic):
    if x == y:
        print('s')
    else:
        print('o')

print('Ethnicity of the TOP2 stranger face id:')    
for x in all_ethnics_top2:
    print(x)
print('TOP2 stranger id same or other race/ethnicity of the participant:')
for x, y in zip(all_ethnics_top2, subjs_ethnic):
    if x == y:
        print('s')
    else:
        print('o')

print('Ethnicity of the TOP3 stranger face id:')   
for x in all_ethnics_top3:
    print(x)
print('TOP3 stranger id same or other race/ethnicity of the participant:')
for x, y in zip(all_ethnics_top3, subjs_ethnic):
    if x == y:
        print('s')
    else:
        print('o')

print('Distance from TOP1 stranger id:') 
for x in all_dist_top1:
    print(round(x, 2))
print('Distance from TOP2 stranger id:') 
for x in all_dist_top2:
    print(round(x, 2))
print('Distance from TOP3 stranger id:')
for x in all_dist_top3:
    print(round(x, 2))
print('Average distance from all own-race stranger ids:')
for x in all_avg_dist_same_ethnic:
    print(round(x, 2))
print('Average distance from all other-race stranger ids:')
for x in all_avg_dist_other_ethnic:
    print(round(x,2))

print('MEDIAN of the TOP1 distance:')
print(round(statistics.median(global_dist_top1), 2))
print('MEDIAN of the TOP2 distance:')
print(round(statistics.median(global_dist_top2), 2))
print('MEDIAN of the TOP3 distance:')
print(round(statistics.median(global_dist_top3),2))
print(round(statistics.median(global_dist_same_ethnic),2))
print(round(statistics.median(global_dist_other_ethnic),2))

global_dist_same_ethnic = [round(x, 2) for x in global_dist_same_ethnic]
print('Median of the global distance same ethnic:')
print(round(statistics.median(global_dist_same_ethnic),2))

global_dist_other_ethnic = [round(x, 2) for x in global_dist_other_ethnic]
print('Median of the global distance other ethnic:')
print(round(statistics.median(global_dist_other_ethnic),2))

print('# of unique stranger identites that are CAU:')
for _ in all_cau_count:
    print(_)

print('# of unique stranger identites that are EA:')
for _ in all_easia_count:
    print(_)

print('# of unique stranger identites that are CALL:')
for _ in all_Call_count:
    print(_)

print('# of unique stranger identites that are AFR:')
for _ in all_afr_count:
    print(_)



