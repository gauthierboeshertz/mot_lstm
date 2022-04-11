

from data_generator import  *
from data_load import *
from PIL import Image


### should be used once to create the data to train the osnet

detections_list,gts_list,frames_list = gen_mot2015_npy()
detections_list16,gts_list16,frames_list16 = gen_mot2016_npy()


frames_list.append(frames_list16[1])
frames_list.append(frames_list16[4])
frames_list.append(frames_list16[5])

gts_list.append(gts_list16[1])
gts_list.append(gts_list16[4])
gts_list.append(gts_list16[5])

max_id =0
frames_freq_per_id= dict()
for gts_list_idx,gts in enumerate(gts_list):
    
    id_counter=0
    gt_by_frame  = groupby(gts,0)
    gt_id_map = dict()
    
    
    
    for frame_num in gt_by_frame:
        gt_in_frame = gt_by_frame[frame_num]
        for idx,gt in enumerate(gt_in_frame):
        
            other_gt  = np.delete(gt_in_frame,idx,0)
            _,iod,_ = overlap(gt,other_gt)
            if np.any(iod>0.5):
                continue
            else :
                im_to_save =get_bb_content_pil(gt,frames_list[gts_list_idx][int(gt[0])-1])
                if gt[1] in gt_id_map:
                    gt_id = gt_id_map[gt[1]]
                else :
                    gt_id_map[gt[1]] = id_counter
                    id_counter +=1
                    gt_id = gt_id_map[gt[1]]
                    
                    
                    
                frame_nbr = frames_freq_per_id.get(gt_id,0)
                frames_freq_per_id[gt_id] = frame_nbr +1
                path = str(int(gt_id)).zfill(3)+'_'+str(int(frame_nbr)).zfill(3)+'.jpg'
                
                im_to_save.save('/media/data/gauthier/mot_cropped/'+str(gts_list_idx)+'/'+path)
        if frame_num%100 ==0:
            print('in frame:', frame_num)
    print(id_counter)