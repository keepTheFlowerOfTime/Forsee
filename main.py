import datetime
import json
import os
import pickle

import fire
import torch
import torch.utils.data as torchdata
import numpy as np

import codesecurity.data.api as data_api
from codesecurity.tasks.code_authorship_attribution import (
    ForseeCachesMetatadata, ForseeSuperParameter, caa_eval, caa_eval_forsee,
    caa_forsee_kfold, caa_forsee_train_test)
from codesecurity.tasks.code_authorship_attribution.caa import (
    caa_DLCAIS_kfold, caa_build_addon_data, caa_build_external_data, caa_eval_forsee_vanille, caa_forsee, caa_forsee_kfold_vanille, caa_measure_forsee,
    label_sample_with_author_repo,caa_forsee_classification)
from codesecurity.tasks.code_authorship_attribution.prepare_torch import (
    ForseeDataset, fold_dataset, prepare_training_set)
from codesecurity.tasks.code_authorship_attribution.preprocessing import (
    DLCAISSuperParameter, build_lexical_tfidfmodule, build_syntactic_tfidfmodule,
    prepare_forsee_features,ForseeFeatureBuilder)
from codesecurity.tasks.code_authorship_attribution.training_model import \
    embeding_model,measure_model


from codesecurity.data.api import pickle_load
from codesecurity.utils.pretty_print import percent

DEVICE_GPU=torch.device('cuda:0')
DEVICE_CPU=torch.device('cpu')

def training_forsee(dataset_dir,sp,k=8,eval_dir=None,mode='kfold',use_caches=True,device='gpu',rebuild=False,partial=None,store_loc=None,**kwargs):
    meta=ForseeCachesMetatadata.auto(dataset_dir,store_loc=store_loc)
    sp=ForseeSuperParameter.instance(sp)
    
    temp_partial=[None,None,None]
    if partial is not None:
        partial=str(partial)
        temp_partial[0]=True if partial[0]=='1' else False
        temp_partial[1]=True if partial[1]=='1' else False
        temp_partial[2]=True if partial[2]=='1' else False

        partial=temp_partial
        print(partial)
    if rebuild:
        if os.path.exists(meta.caches_dir):
            files=os.listdir(meta.caches_dir)
            for f in files:
                os.remove(os.path.join(meta.caches_dir,f))
                print(f"remove {f} in {meta.caches_dir}")
    if device.strip()=='gpu':
        device=DEVICE_GPU
    else:
        device=DEVICE_CPU
        
    
    if mode=='kfold':
        caa_forsee_kfold(dataset_dir,None,sp,k,device,meta=meta,use_caches=use_caches,partial=partial)

    elif mode=='std':
        caa_forsee_train_test(dataset_dir,eval_dir,None,sp,device,meta=meta,use_caches=use_caches)

def training_forsee_vanille(dataset_dir,sp,k=8,eval_dir=None,mode='kfold',use_caches=True,device='gpu',rebuild=False,**kwargs):
    meta=ForseeCachesMetatadata.auto(dataset_dir)
    sp=ForseeSuperParameter.instance(sp)
    
    if rebuild:
        if os.path.exists(meta.caches_dir):
            files=os.listdir(meta.caches_dir)
            for f in files:
                os.remove(os.path.join(meta.caches_dir,f))
                print(f"remove {f} in {meta.caches_dir}")
    if device.strip()=='gpu':
        device=DEVICE_GPU
    else:
        device=DEVICE_CPU
        
    
    if mode=='kfold':
        caa_forsee_kfold_vanille(dataset_dir,None,sp,k,device,meta=meta,use_caches=use_caches)

    elif mode=='std':
        caa_forsee_train_test(dataset_dir,eval_dir,None,sp,device,meta=meta,use_caches=use_caches)

def training_DLCAIS(dataset_dir,k=8,eval_dir=None,mode='kfold',use_caches=True,device='gpu',rebuild=False):
    meta=ForseeCachesMetatadata.auto(dataset_dir)
    sp=DLCAISSuperParameter.default()
    
    if rebuild:
        if os.path.exists(meta.DLCAIS):
            os.remove(meta.DLCAIS)
            print(f"remove {meta.DLCAIS} in {meta.caches_dir}")
                
    if device.strip()=='gpu':
        device=DEVICE_GPU
    else:
        device=DEVICE_CPU
        
    
    caa_DLCAIS_kfold(dataset_dir,None,sp,k,device,meta=meta,use_caches=use_caches)


def eval_forsee(dataset_dir,sp,prefer=None,mode='kfold',k=8,device='cpu',include_submodels=True,**kwargs):

    if prefer is None:
        prefer=[1.,1.,1.]
        # 1,1,1.5
        # 1.5,1,1.5
        # 3,1,3

    meta=ForseeCachesMetatadata.auto(dataset_dir)
    sp=ForseeSuperParameter.instance(sp)

    if device=='gpu':
        device=DEVICE_GPU
    else:
        device=DEVICE_CPU

    if mode=='kfold':
        
        full_dataset=pickle_load(meta.training_refine_data_file)
        full_dataset=ForseeDataset(full_dataset)
        class_number=full_dataset.class_number
        _,eval_data=fold_dataset(full_dataset,k,42)
    else:
        eval_data=pickle_load(meta.test_refine_data_file)
        eval_data=ForseeDataset(eval_data)
        class_number=eval_data.class_number
        
    #print(eval_data[0])

    raw_prec=caa_eval_forsee(meta,sp,class_number,eval_data,device,prefer)
    
    if include_submodels:
    
        independent_model_precs=caa_eval(meta,sp,class_number,eval_data,device)
        
        
        independent_model_precs.insert(0,raw_prec)
        for i in range(len(independent_model_precs)):
            print(f"{percent(independent_model_precs[i])}",end='\t')
        print()

def eval_prefer(dataset_dir,sp,k=8,test_layout=True,test_lexical=False,test_syntactic=True,device='cpu',**kwargs):
    # python main.py eval_prefer data/a_github_c github_c --k=10 --device=gpu
    count=10
    layout_step=0.3
    lexical_step=0.
    syntactic_step=0.5
    base_prefer=[1.,1.,1.]
    
    for i in range(count):
        if test_layout:
            base_prefer[0]+=layout_step
        if test_lexical:
            base_prefer[1]+=lexical_step
        if test_syntactic:
            base_prefer[2]+=syntactic_step        
        eval_forsee(dataset_dir,sp,prefer=base_prefer,mode='kfold',k=k,device=device,include_submodels=False)
        eval_external_data_forsee(dataset_dir,'generate_data/a_github_c/program_file/untargeted_attack_file','model_caches/a_github_c_untarget',sp,device=device,use_caches=True,prefer=base_prefer,include_submodels=False)
        print('-----')
def eval_forsee_vanille(dataset_dir,sp,mode='kfold',k=8,device='cpu',**kwargs):
    meta=ForseeCachesMetatadata.auto(dataset_dir)
    sp=ForseeSuperParameter.instance(sp)

    if device=='gpu':
        device=DEVICE_GPU
    else:
        device=DEVICE_CPU

    if mode=='kfold':
        
        full_dataset=pickle_load(meta.training_refine_data_file)
        full_dataset=ForseeDataset(full_dataset)
        class_number=full_dataset.class_number
        _,eval_data=fold_dataset(full_dataset,k,42)
    else:
        eval_data=pickle_load(meta.test_refine_data_file)
        eval_data=ForseeDataset(eval_data)
        class_number=eval_data.class_number
        
    #print(eval_data[0])

    raw_prec=caa_eval_forsee_vanille(meta,sp,class_number,eval_data,device)
    print()


def eval_external_data_forsee(dataset_dir,external_dir,caches_dir,sp,prefer=None,device='cpu',use_caches=True,include_submodels=True,**kwargs):
    if prefer is None:
        prefer=[1.5,1,3]
        # 1,1,1.5
        # 1.5,1,1.5
        # 3,1,3
    meta=ForseeCachesMetatadata.auto(dataset_dir)
    sp=ForseeSuperParameter.instance(sp)
    
    if device=='gpu':
        device=DEVICE_GPU
    else:
        device=DEVICE_CPU

    #full_dataset=torch.load(meta.training_default_task_data_file)
    data=caa_build_addon_data(external_dir,caches_dir,meta,sp,use_caches=use_caches)
    
    caa_eval_forsee(meta,sp,data.class_number,data,device,prefer)
    if include_submodels:
        caa_eval(meta,sp,data.class_number,data,device)

def eval_external_data_forsee_vanille(dataset_dir,external_dir,caches_dir,sp,device='cpu',use_caches=True,**kwargs):
    meta=ForseeCachesMetatadata.auto(dataset_dir)
    sp=ForseeSuperParameter.instance(sp)
    
    if device=='gpu':
        device=DEVICE_GPU
    else:
        device=DEVICE_CPU

    #full_dataset=torch.load(meta.training_default_task_data_file)
    data=caa_build_addon_data(external_dir,caches_dir,meta,sp,use_caches=use_caches)
    
    caa_eval_forsee_vanille(meta,sp,data.class_number,data,device)

def validate_forsee(dataset_dir,sp,target_caches_file=None,mode='kfold',k=8,device='cpu',reverse=False):
    meta=ForseeCachesMetatadata.auto(dataset_dir)
    sp=ForseeSuperParameter.instance(sp)

    if device=='gpu':
        device=DEVICE_GPU
    else:
        device=DEVICE_CPU

    if target_caches_file is not None:
        mode='std'

    if mode=='kfold':
        if target_caches_file is None: target_caches_file=meta.training_refine_data_file
        full_dataset=pickle_load(target_caches_file)
        full_dataset=ForseeDataset(full_dataset)
        class_number=full_dataset.class_number
        _,eval_data=fold_dataset(full_dataset,k,42)
    else:
        if target_caches_file is None: target_caches_file=meta.test_refine_data_file
        eval_data=pickle_load(target_caches_file)
        eval_data=ForseeDataset(eval_data)
        class_number=eval_data.class_number
        
    #print(eval_data[0])

    error_stat,error_times=caa_forsee_classification(meta,sp,class_number,eval_data,device,reverse)
    total_error_number=0
    for e in zip(*error_stat):
        if sum(e)>0.1:
            print(e)
        if min(e)>0.01:
            total_error_number+=1

    print(percent(total_error_number/len(error_stat[0])))
    #print(error_stat[1:])
    #print(error_times[1:])

def eval_robust_forsee(dataset_dir,external_dir,caches_dir,sp,device='cpu',use_caches=True,mode='untargeted'):
    meta=ForseeCachesMetatadata.auto(dataset_dir)
    sp=ForseeSuperParameter.instance(sp)
    
    if device=='gpu':
        device=DEVICE_GPU
    else:
        device=DEVICE_CPU
    
    #full_dataset=torch.load(meta.training_default_task_data_file)
    data=caa_build_addon_data(external_dir,caches_dir,meta,sp,use_caches=use_caches)
    
    class_number=get_class_number(dataset_dir)
    
    pcs=caa_eval_forsee(meta,sp,class_number,data,device)
    submodels_pcs=caa_eval(meta,sp,class_number,data,device)

    if mode=='untargeted':
        pcs=1-pcs
        submodels_pcs=[1-p for p in submodels_pcs]

    print(f"{mode} attack success rate: {pcs}")
    print(f"{mode} attack success rate for submodels: {submodels_pcs}")

def measurce_forsee(dataset_dir,sp,k=10,device='cpu',out_file=None):
    if device=='gpu':
        device=DEVICE_GPU
    else:
        device=DEVICE_CPU
        
    if out_file is None:
        out_file=f'temp/{sp}.pt'

    meta=ForseeCachesMetatadata.auto(dataset_dir)
    sp=ForseeSuperParameter.instance(sp)

    full_dataset=pickle_load(meta.training_refine_data_file)
    full_dataset=ForseeDataset(full_dataset)
    
    caa_measure_forsee(sp,full_dataset,device,k=k,out_file=out_file)

def weight_view(dataset_dir,sp,device='cpu'):
    sp_name=sp
    meta=ForseeCachesMetatadata.auto(dataset_dir)
    sp=ForseeSuperParameter.instance(sp)

    if device=='gpu':
        device=DEVICE_GPU
    else:
        device=DEVICE_CPU

    full_dataset=pickle_load(meta.training_refine_data_file)
    full_dataset=ForseeDataset(full_dataset)
    class_number=full_dataset.class_number
    

    model=caa_forsee(meta,sp,class_number,device)
    weights=model.weights().detach().cpu().numpy()

    weights=np.abs(weights)
    weights=np.sum(weights,axis=0)
    
    weights=weights.tolist()

    with open(f'temp/{sp_name}.txt','w') as f:
        f.write(weights.__str__()) 

    print(weights)

def get_class_number(dataset_dir):
    meta=ForseeCachesMetatadata.auto(dataset_dir)
    group=data_api.pickle_load(meta.training_raw_data_file)
    return len(group.addon)

def build_training_caches(dataset_dir,sp,min_number=1,max_number=2**32-1,rebuild=False,store_loc=None):
    meta=ForseeCachesMetatadata.auto(dataset_dir,store_loc=store_loc)
    sp=ForseeSuperParameter.instance(sp)
    raw_data_path=meta.training_raw_data_file

    if os.path.exists(raw_data_path) and not rebuild:
        print(f"caches already exists in {raw_data_path}")

    training_set=prepare_training_set(dataset_dir,None,None,meta,sp)
    
    #refine_features=prepare_forsee_features(dataset_dir,meta,sp,min_number=min_number,max_number=max_number)
    #refine_features.save(meta.training_refine_data_file)
    print(f"caches build for {len(training_set.features.id_mapping)} authors.")

def get_file_embeding(meta_dir,input_dir,caches_file,sp,device,out_dir=None,mode='repo'):
    def deduce_url(absolute_path):
        start_index = absolute_path.find("cpp/") + len("cpp/")
        segment1 = absolute_path[start_index:]

        return segment1

    meta=ForseeCachesMetatadata.auto(meta_dir)
    sp=ForseeSuperParameter.instance(sp)

    if device=='gpu':
        device=DEVICE_GPU
    else:
        device=DEVICE_CPU

    list_author_handle=None
    if mode=='repo' or mode=='test':
        list_author_handle=label_sample_with_author_repo

    class_number=get_class_number(meta_dir)

    if meta_dir==input_dir:
        external_data=prepare_training_set(meta_dir,None,None,meta,sp)
    else:
        external_data=caa_build_external_data(input_dir,caches_file,meta,sp,list_author_handle=list_author_handle)
    
    model=caa_forsee(meta,sp,class_number,device)


    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    

    max_row_per_file=5000
    counter=0
    i=0
    group_no=0

    buffer=[]

    for embeding,labels in embeding_model(model,external_data):
        create_time=datetime.datetime.now()
        update_time=create_time
        for one_embeding in embeding:
            one_embeding=one_embeding.tolist()
            url=external_data.features.samples[i].origin_paths
            
            row_dict={
                'url':deduce_url(url),
                'create_time':create_time.__str__(),
                'update_time':update_time.__str__(),
                'vector':one_embeding,
                'type':'file'
            }

            buffer.append(row_dict)

            counter+=1
            i+=1

        if counter>=max_row_per_file:
            counter=0
            if out_dir:
                json_path=os.path.join(out_dir,f'embeding{group_no}.json')
                with open(json_path,'w') as f:
                    if mode=='test':
                        json.dump(buffer,f,indent=4)
                    else:
                        json.dump(buffer,f)

            group_no+=1
            buffer=[]

            if mode=='test':
                break

def get_repo_embeding(meta_dir,input_dir,caches_file,sp,device,out_dir=None,mode='repo',label="",min_number=1,max_number=2**32-1):
    def deduce_url(absolute_path):
        start_index = absolute_path.find("cpp/") + len("cpp/")
        segment1 = absolute_path[start_index:]
        start_index = absolute_path.find("cpp/") + len("cpp/")
        end_index = absolute_path.find("/", start_index)
        segment1 = absolute_path[start_index:end_index]

        # 提取 ctf_tools
        start_index = absolute_path.find(segment1) + len(segment1) + 1
        end_index = absolute_path.find("/", start_index)
        segment2 = absolute_path[start_index:end_index]

        return f'{segment1}/{segment2}'

    meta=ForseeCachesMetatadata.auto(meta_dir)
    sp=ForseeSuperParameter.instance(sp)

    if device=='gpu':
        device=DEVICE_GPU
    else:
        device=DEVICE_CPU

    list_author_handle=None
    if mode=='repo' or mode=='test':
        list_author_handle=label_sample_with_author_repo

    class_number=get_class_number(meta_dir)
    external_data=None

    if meta_dir==input_dir:
        external_data=prepare_training_set(meta_dir,None,None,meta,sp)
    else:
        external_data=caa_build_external_data(input_dir,caches_file,meta,sp,list_author_handle=list_author_handle,min_number=min_number,max_number=max_number)
    
    model=caa_forsee(meta,sp,class_number,device)


    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    
    max_file=5000 if mode=='test' else 1000000
    i=0

    repos={}

    for embeding,labels in embeding_model(model,external_data):
        create_time=datetime.datetime.now()
        update_time=create_time
        for one_embeding in embeding:
            url=external_data.features.samples[i].origin_paths
            
            repo_url=deduce_url(url)
            if repo_url not in repos:
                repos[repo_url]={
                'url':deduce_url(url),
                'create_time':create_time.__str__(),
                'update_time':update_time.__str__(),
                'vector':one_embeding,
                'layout_vec':one_embeding[:sp.lay_hidden_dim],
                'lexical_vec':one_embeding[sp.lay_hidden_dim:sp.lay_hidden_dim+sp.lex_hidden_dim],
                'syntactic_vec':one_embeding[sp.lay_hidden_dim+sp.lex_hidden_dim:],
                'type':'repo'
            }
            else:
                repos[repo_url]['vector']+=one_embeding
                repos[repo_url]['layout_vec']+=one_embeding[:sp.lay_hidden_dim]
                repos[repo_url]['lexical_vec']+=one_embeding[sp.lay_hidden_dim:sp.lay_hidden_dim+sp.lex_hidden_dim]
                repos[repo_url]['syntactic_vec']+=one_embeding[sp.lay_hidden_dim+sp.lex_hidden_dim:]

            i+=1

        if i>=max_file:
            break

    if out_dir:
        json_path=os.path.join(out_dir,f'repo_embeding{label}.json')
        for e in repos:
            v=repos[e]['vector']
            v=v/np.linalg.norm(v)
            repos[e]['vector']=v.tolist()
            
            v=repos[e]['lexical_vec']
            v=v/np.linalg.norm(v)
            repos[e]['lexical_vec']=v.tolist()

            v=repos[e]['layout_vec']
            v=v/np.linalg.norm(v)
            repos[e]['layout_vec']=v.tolist()
            
            v=repos[e]['syntactic_vec']
            v=v/np.linalg.norm(v)
            repos[e]['syntactic_vec']=v.tolist()

        repos=list(repos.values())

        with open(json_path,'w') as f:
            if mode=='test':
                json.dump(repos,f,indent=4)
            else:
                json.dump(repos,f)


#python main.py measurce_forsee data/gcj_cpp gcj_cpp --k=8 --device=gpu
#python main.py eval_external_data_forsee data/gcj_cpp_robust/train data/gcj_cpp_unuse_remove/test model_caches/gcj_cpp_robust_test gcj_cpp --device=gpu
#python main.py training_forsee data/gcj_cpp_robust/train gcj_cpp --eval_dir=data/gcj_cpp_robust/validate --mode=std --device=gpu
#python main.py weight_view data/gcj_cpp gcj_cpp --device=gpu
#python main.py training_DLCAIS data/gcj_cpp --k=8 --device=gpu --rebuild=True
#python main.py validate_forsee data/gcj_cpp gcj_cpp --k=8 --device=gpu
#python main.py validate_forsee data/a_github_c github_c --k=10 --device=gpu
#python main.py validate_forsee data/a_github_c github_c --device=gpu --target_caches_file=model_caches/a_github_c_target/refine_data.pt
#python main.py get_repo_embeding /mnt/XiaoweiGuo/data/Malicious500Group/cpp /mnt/XiaoweiGuo/data/Malicious500Group/cpp /temp/test/data.pt persp_cpp gpu temp/mongo_output/ --mode==test
#python main.py get_file_embeding /mnt/XiaoweiGuo/data/Malicious500Group/cpp /mnt/XiaoweiGuo/data/Malicious500Group/cpp /temp/test/data.pt persp_cpp gpu temp/mongo_output/
#python main.py build_training_caches /mnt/XiaoweiGuo/data/Malicious500Group/cpp persp_cpp --min_number=10 --max_number=1000
#python main.py training_forsee data/gcj_cpp gcj_cpp --k=8 --device=gpu --partial=011
#python main.py eval_external_data_forsee data/gcj_cpp generate_data/gcj_cpp/program_file/targeted_attack_file model_caches gcj_cpp --device=gpu
# def build_word_caches(dataset_dir,rebuild=False):
#     meta=ForseeCachesMetatadata.auto(dataset_dir)
#     raw_data_path=meta.training_raw_data_file
#     if not os.path.exists(raw_data_path):
#         print(f"caches not exists in {raw_data_path}, exit.")

#     if rebuild:
#         if os.path.exists(meta.lexical_file):
#             os.remove(meta.lexical_file)
#         if os.path.exists(meta.syntactic_file):
#             os.remove(meta.syntactic_file)
    
#     with open(raw_data_path,'rb') as f:
#         raw_features=pickle.load(f)

#     print("loading caches complete.")

#     build_lexical_tfidfmodule(raw_features,meta.lexical_file)

#     print('build lexical module finish.')

#     build_syntactic_tfidfmodule(raw_features,meta.syntactic_file)

#     print('build syntactic module finish.')
#________________________________

def migrate_core_author(origin_dir,new_dir):
    authors=read_core_author_set(origin_dir)
    data_migration(authors,new_dir)

def abstract(dataset_dir):
    authors=read_author_set(dataset_dir)
    file_number=sum([author.file_number for author in authors])
    total_size=sum([author.size for author in authors])

    file_number_range=[]

    for author in authors:
        file_number_range.append(author.file_number)
        print(f'{author.author_info.login}/{author.file_number}')

    file_number_range.sort(reverse=True)

    print(f'file number: {file_number}')
    print(f'total size: {round(total_size/(2**20),4)}MB')
    print(f'top100/top50 file number: {file_number_range[99]}/{file_number_range[49]}')

if __name__=="__main__":
    fire.Fire()