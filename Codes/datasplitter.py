import numpy as np
from sklearn.utils import shuffle
import os
import sys
import glob


class TrainValDataSplitter(object):
    
    
    def __init__(self,data_home = '../Data/', dataset = '', seqs = None,split = 0.7,random_seed = None,
                params = None):
        self.data_home = data_home
        self.split_rate = split
        self.seqs = seqs
        
        
        self.nSkips_row = params['nSkips_r']
        self.nSkips_col = params['nSkips_c']        
        
            
        if random_seed == None:
            self.shuffle_flag = False
        else:
            self.shuffle_flag = True
            self.random_seed = random_seed
        
        if dataset == 'KITTI':
            self.dataset_name = 'KITTI_Depth'
            self.dataset_catgry = 1
            train_datalist, test_datalist = self.init_KITTI_DataPath(self.seqs, params)
        
        else:
            RuntimeError('Dataset cannot be recognized. Exiting ..!')
            sys.exit(0)
        self.training_tags = []
        self.validation_tags = []
        
        
        if self.dataset_catgry == 1:
            traindata_tags = self.integrate_dataset_KITTI(train_datalist, params)
            testdata_tags = self.integrate_dataset_KITTI(test_datalist, params, val_set = params['val_set'])

            self.training_tags,self.validation_tags = self.create_datasplit_kitti(traindata_tags,self.split_rate)
            self.training_nexamp = len(self.training_tags)
            self.validation_nexamp = len(self.validation_tags)
            self.test_tags,__ = self.create_datasplit_kitti(testdata_tags,1)
            self.test_nexamp = len(self.test_tags)        
    
    def init_KITTI_DataPath(self,seqs,params):
        train_datalist = []
        test_datalist = []

        if params['Uniformflag']:
            subsample_name = 'Uniform_d%d'%(params['Uniform_samp'])
        else:    
            subsample_name = '%dx%d_nSKips'%(params['nSkips_r'], params['nSkips_c'])
        
        
        Train_Dir = 'train'
            
        if params['val_set'] == 'shortened':
            Test_Dir = 'val_shortened'
        elif params['val_set'] == 'augmented':
            Test_Dir = 'val' 
        else:
            ValueError('Unknown validation set. Exiting ...')
            
             
        self.color_dir = 'color'
        self.fused_gt_dir = 'fused'
        
        self.annotated_gt_dir = 'groundtruth'
        self.sparse_subsample_dir = subsample_name
        
        self.extra_path = 'proj_depth'

        drive_seqs = os.listdir(os.path.join(self.data_home, self.dataset_name, Train_Dir))
        drive_seqs.sort()

        for drive_seq in drive_seqs:

            for cam_id in params['cam_set']:    
                train_path = os.path.join(self.data_home, self.dataset_name, Train_Dir, drive_seq, self.extra_path, self.annotated_gt_dir,'image_%02d'%cam_id)
            
                if not os.path.exists:
                    continue            
            
                train_filenames = glob.glob1(train_path,'*.png')
                train_filenames.sort()
                bagdict = {}
                bagdict['drive_seqs'] = drive_seq
                bagdict['cam_name'] = 'image_%02d'%cam_id    
                bagdict['complete_path'] = os.path.join(self.data_home,self.dataset_name,Train_Dir)                
                bagdict['nfiles'] = train_filenames
                train_datalist.append(bagdict)

        if params['val_set'] == 'shortened':
            test_path = os.path.join(self.data_home, self.dataset_name, Test_Dir, self.extra_path, self.annotated_gt_dir)
            test_filenames = glob.glob1(test_path,'*.png')
            test_filenames.sort()
            bagdict = {}                
            bagdict['complete_path'] = os.path.join(self.data_home, self.dataset_name, Test_Dir)
            bagdict['val_set'] = 'shortened'            
            bagdict['nfiles'] = test_filenames
            test_datalist.append(bagdict)

        else:            
            drive_seqs = os.listdir(os.path.join(self.data_home, self.dataset_name, Test_Dir))
            drive_seqs.sort()

            for drive_seq in drive_seqs:

                for idx, cam_id in enumerate(params['cam_set']):
                    test_path = os.path.join(self.data_home, self.dataset_name, Test_Dir, drive_seq, self.extra_path, 
                                            self.annotated_gt_dir,'image_%02d'%cam_id)
                    if not os.path.exists(test_path):
                        continue
                    
                    test_filenames = glob.glob1(test_path,'*.png')
                    test_filenames.sort()
                    bagdict = {}
                    bagdict['drive_seqs'] = drive_seq
                    bagdict['cam_name'] = 'image_%02d'%cam_id    
                    bagdict['complete_path'] = os.path.join(self.data_home, self.dataset_name, Test_Dir)
                    bagdict['val_set'] = 'augmented'
                    bagdict['nfiles'] = test_filenames
                    test_datalist.append(bagdict)

        return train_datalist, test_datalist        
    
    def integrate_dataset_KITTI(self, datalist, params, val_set = None):
        
        data_tag = []

        
        if val_set == 'shortened':

            

            for ii in range(len(datalist)):    
                for jj in range(len(datalist[ii]['nfiles'])):
                    data_tagtemp = {}

                    data_tagtemp['sparsesubsample_tag'] = os.path.join(datalist[ii]['complete_path'],self.extra_path, self.sparse_subsample_dir,
                                                                        datalist[ii]['nfiles'][jj])                        
                    data_tagtemp['fusedgt_tag'] = os.path.join(datalist[ii]['complete_path'], self.extra_path,self.fused_gt_dir,
                                                                datalist[ii]['nfiles'][jj])
                    data_tagtemp['annotatedgt_tag'] = os.path.join(datalist[ii]['complete_path'], self.extra_path,self.annotated_gt_dir,
                                                                datalist[ii]['nfiles'][jj])                        
                    data_tagtemp['color_tag'] = os.path.join(datalist[ii]['complete_path'],'image',datalist[ii]['nfiles'][jj])

                    data_tag.append(data_tagtemp)

        else:
                    
            for ii in range(len(datalist)):
                
                for jj in range(len(datalist[ii]['nfiles'])):
                    data_tagtemp = {}
                    data_tagtemp['sparsesubsample_tag'] = os.path.join(datalist[ii]['complete_path'], datalist[ii]['drive_seqs'],
                                                            self.extra_path,self.sparse_subsample_dir, datalist[ii]['cam_name'],
                                                            datalist[ii]['nfiles'][jj])                    
                    data_tagtemp['fusedgt_tag'] = os.path.join(datalist[ii]['complete_path'], datalist[ii]['drive_seqs'],
                                                            self.extra_path,self.fused_gt_dir, datalist[ii]['cam_name'],
                                                            datalist[ii]['nfiles'][jj])
                    data_tagtemp['annotatedgt_tag'] = os.path.join(datalist[ii]['complete_path'], datalist[ii]['drive_seqs'],
                                                            self.extra_path, self.annotated_gt_dir, datalist[ii]['cam_name'],
                                                            datalist[ii]['nfiles'][jj])                    
                    data_tagtemp['color_tag'] = os.path.join(datalist[ii]['complete_path'],datalist[ii]['drive_seqs'],
                                                            self.color_dir, datalist[ii]['cam_name'], datalist[ii]['nfiles'][jj])    

                    data_tag.append(data_tagtemp)



        return data_tag    
    
    
    def create_datasplit_kitti(self, data_tags, split_rate, shuffle_flag = False, take_nvaldata = None):

        if shuffle_flag:
            data_tags = shuffle(data_tags, random_state = self.random_seed)

        split_tags_1 = data_tags[:int(round(split_rate*len(data_tags)))]
        split_tags_2 = data_tags[int(round(split_rate*len(data_tags)))+1:]

        if shuffle_flag:
            split_tags_1 = shuffle(split_tags_1, random_state = self.random_seed) 
            split_tags_2 = shuffle(split_tags_2, random_state = self.random_seed)
        
        if take_nvaldata is not None:
            split_tags_1 = split_tags_1[:take_nvaldata]

        return split_tags_1, split_tags_2


if __name__== '__main__':

    synthia_seqs = ['SYNTHIA-SEQS-02-SUMMER','SYNTHIA-SEQS-04-SUMMER']
    TrainValDataSplitter(dataset = 'Synthia',seqs = synthia_seqs)
    KITTI_seqs = ['2011_09_26','2011_09_28','2011_09_29','2011_09_30','2011_10_03']
    TrainValDataSplitter(dataset = 'KITTI',seqs = KITTI_seqs)
