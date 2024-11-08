from __future__ import annotations
import os
import pickle
import random
from codesecurity.data.package_extract import iter_dir
from codesecurity.data.caches_manager import GroupPipe

def label_sample_with_dirname(dir,label=None):

    classes=os.listdir(dir)
    classes=[e for e in classes if os.path.isdir(os.path.join(dir,e))]

    return [(os.path.join(dir,e),e) for e in classes]

def list_file_in_dir(dir,max_size=2**21):
    files=[os.path.join(dir,e) for e in os.listdir(dir) if os.path.isfile(os.path.join(dir,e))]

    return [e for e in files if os.stat(e).st_size<=max_size]

def list_all_file_in_dir(dir,max_size=2**21):
    files=iter_dir(dir)

    return [e for e in files if os.stat(e).st_size<=max_size]

def filter_file(dir,filters:list[str]=[],max_size=2**21):
    
    all_files=list_all_file_in_dir(dir,max_size=max_size)
    
    result=[]
    for file in all_files:
        for filter in filters:
            if file.endswith(filter):
                result.append(file)
                break
            
    return result

def list_dataset(dataset_dir,label_handle=None):
    if label_handle is None: label_handle=label_sample_with_dirname
    sample_groupby_labels=[(label,list_all_file_in_dir(e)) for e,label in label_handle(dataset_dir)]

    for sample_label,files in sample_groupby_labels:
        yield files,sample_label            

def split_dataset(dataset_dir,train_dir,validate_dir,test_dir,ratio=[0.8,0.1,0.1],label_handle=None,seed=None):
    def spilt_files(files,ratio=[0.8,0.1,0.1]):
        file_number=len(files)
        test_number=max(1,int(file_number*ratio[2])) if ratio[2]>0 else 0
        validate_number=max(1,int(file_number*ratio[1])) if ratio[1]>0 else 0
        train_number=file_number-test_number-validate_number
        random.shuffle(files)
        
        return files[:train_number],files[train_number:train_number+validate_number],files[train_number+validate_number:]
    
    if seed is not None:
        random.seed(seed)

    if label_handle is None: label_handle=label_sample_with_dirname
    sample_groupby_labels=[(label,list_all_file_in_dir(e)) for e,label in label_handle(dataset_dir)]

    for sample_label,files in sample_groupby_labels:
        train_files,validate_files,test_files=spilt_files(files,ratio=ratio)
        for file in train_files:
            file_name=os.path.split(file)[-1]
            target_file=os.path.join(train_dir,sample_label,file_name)
            os.makedirs(os.path.split(target_file)[0],exist_ok=True)
            os.rename(file,target_file)
        for file in validate_files:
            file_name=os.path.split(file)[-1]
            target_file=os.path.join(validate_dir,sample_label,file_name)
            os.makedirs(os.path.split(target_file)[0],exist_ok=True)
            os.rename(file,target_file)
        for file in test_files:
            file_name=os.path.split(file)[-1]
            target_file=os.path.join(test_dir,sample_label,file_name)
            os.makedirs(os.path.split(target_file)[0],exist_ok=True)
            os.rename(file,target_file)

def split_balance_dataset(dataset_dir,out_dirs,ratios,label_handle=None):
    pass




def pickle_load(path):
    with open(path,'rb') as f:
        return pickle.load(f)
    
def pickle_save(obj,path):
    with open(path,'wb') as f:
        pickle.dump(obj,f)