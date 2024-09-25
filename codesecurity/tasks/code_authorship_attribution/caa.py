import codesecurity.tasks.code_authorship_attribution.preprocessing as preprocessing
import codesecurity.tasks.code_authorship_attribution.prepare_torch as prepare_torch
import codesecurity.tasks.code_authorship_attribution.caches_manager as caches_manager
import codesecurity.tasks.code_authorship_attribution.training_model as training_model
import codesecurity.tasks.code_authorship_attribution.model as caa_model
import codesecurity.data.api as dataapi
import torch
import os


def label_sample_with_author_repo(dir,label=None):

    authors=os.listdir(dir)
    authors=[e for e in authors if os.path.isdir(os.path.join(dir,e))]
    
    all_repos=[]
    for author in authors:
        author_repos=os.listdir(os.path.join(dir,author))
        author_repos=[(author,e) for e in author_repos if os.path.isdir(os.path.join(dir,author,e))]
        all_repos+=author_repos

    return [(os.path.join(dir,a,r),f'{a}/{r}') for a,r in all_repos]

def caa_forsee_kfold(dataset_dir,lang,sp,k,device,use_caches=True,meta=None,list_author_handle=None,partial=None):
    if meta is None:
        meta=caches_manager.ForseeCachesMetatadata.auto(dataset_dir,"model_caches")
    
    training_data,test_data,id_mapping=prepare_torch.prepare_fold_set(dataset_dir,lang,list_author_handle,meta,sp,k)
    
    class_number=len(id_mapping)
    
    if partial is None:
        forsee,independence_models=training_model.train_forsee(training_data,test_data,class_number,meta,sp,device,use_caches)
    else:
        use_layout,use_lexical,use_syntactic=partial
        forsee=training_model.caa_patial_forsee_training(training_data,test_data,meta,sp,class_number,device,use_layout,use_lexical,use_syntactic)
        independence_models=[]
    return forsee,independence_models

def caa_forsee_kfold_vanille(dataset_dir,lang,sp,k,device,use_caches=True,meta=None,list_author_handle=None):
    if meta is None:
        meta=caches_manager.ForseeCachesMetatadata.auto(dataset_dir,"model_caches")
    
    training_data,test_data,id_mapping=prepare_torch.prepare_fold_set(dataset_dir,lang,list_author_handle,meta,sp,k)
    
    class_number=len(id_mapping)
    
    forsee=training_model.caa_vanille_forsee_training(training_data,test_data,meta,sp,class_number,device)

    return forsee

def caa_forsee_train_test(training_dir,test_dir,lang,sp,device,use_caches=True,meta=None,list_author_handle=None):
    if meta is None:
        meta=caches_manager.ForseeCachesMetatadata.auto(training_dir,"model_caches")
    
    training_data=prepare_torch.prepare_training_set(training_dir,lang,list_author_handle,meta,sp)
    id_mapping=training_data.features.id_mapping
    test_data=prepare_torch.prepare_eval_set(test_dir,lang,list_author_handle,id_mapping,meta,sp)
    

    class_number=len(id_mapping)
    
    forsee,indepence_models=training_model.train_forsee(training_data,test_data,class_number,meta,sp,device,use_caches)
    
    return forsee,indepence_models

def caa_build_addon_data(dataset_dir,caches_dir,meta,sp,list_author_handle=None,use_caches=True):
    level1,level2,level3=caches_manager.ForseeCachesMetatadata.external_caches(caches_dir)
    lexical_file=meta.lexical_file
    syntactic_file=meta.syntactic_file
    id_mapping=dataapi.pickle_load(meta.training_raw_data_file).addon
    os.makedirs(caches_dir,exist_ok=True)
    
    if not use_caches:
        if os.path.exists(level1):
            os.remove(level1)
        if os.path.exists(level2):
            os.remove(level2)
        if os.path.exists(level3):
            os.remove(level3)
    
    return prepare_torch.prepare_subdataset(dataset_dir,None,list_author_handle,id_mapping,sp,level3,level1,level2,lexical_file,syntactic_file)

def caa_build_external_data(dataset_dir,caches_file,meta:caches_manager.ForseeCachesMetatadata,sp:preprocessing.ForseeSuperParameter,use_label=False,list_author_handle=None,min_number=1,max_number=2**32-1):
    id_mapping=None
    if use_label:
        group=dataapi.pickle_load(meta.training_raw_data_file)
        id_mapping=group.addon
        

    dataset=prepare_torch.prepare_external_data(dataset_dir,caches_file,list_author_handle,id_mapping,sp,meta.lexical_file,meta.syntactic_file,min_number,max_number)

    return dataset
    
def caa_DLCAIS_kfold(dataset_dir,lang,sp,k,device,use_caches=True,meta=None,list_author_handle=None):
    if meta is None:
        meta=caches_manager.ForseeCachesMetatadata.auto(dataset_dir,"model_caches")
    
    training_data=prepare_torch.prepare_training_set(dataset_dir,lang,list_author_handle,meta,sp)
    
    DLCAIS_dataset=prepare_torch.DLCAISDataset(training_data.features,sp)
    
    training_set,eval_set=prepare_torch.fold_dataset(DLCAIS_dataset,k,42)
    
    class_number=len(training_data.features.id_mapping)
    
    DLCAIS_model=training_model.train_DLCAIS_nn(training_set,eval_set,class_number,meta,sp,device,use_caches)
    
    classfier=training_model.train_DLCAIS_RFC(training_set,eval_set,class_number,meta,sp,DLCAIS_model,device)
    
    return DLCAIS_model,classfier

def caa_eval_forsee_vanille(meta:caches_manager.ForseeCachesMetatadata,sp,class_number,data,device):
    model=caa_forsee_vanille(meta,sp,class_number,device)
    
    return training_model.eval_forsee(model,data,device)

def caa_eval_forsee(meta:caches_manager.ForseeCachesMetatadata,sp,class_number,data,device,prefer=None):
    model=caa_forsee(meta,sp,class_number,device,prefer=prefer)
    
    return training_model.eval_forsee(model,data,device)
    
def caa_eval(meta:caches_manager.ForseeCachesMetatadata,sp,class_number,data,device):
    layout_model,lexical_model,syntactic_model=caa_model.prepare_independence_model(sp,device,class_number)
    
    layout_parameters=torch.load(meta.layout_extractor)
    lexical_parameters=torch.load(meta.lexical_extractor)
    syntactic_parameters=torch.load(meta.syntactic_extractor)
    
    layout_model.load_state_dict(layout_parameters)
    lexical_model.load_state_dict(lexical_parameters)
    syntactic_model.load_state_dict(syntactic_parameters)
    
    layout_data=prepare_torch.ForseeLayoutDataset(data)
    lexical_data=prepare_torch.ForseeLexicalDataset(data)
    syntactic_data=prepare_torch.ForseeSyntacticDataset(data)
    
    pcs_layout=training_model.eval_indepence_model(layout_model,layout_data,device)
    pcs_lexical=training_model.eval_indepence_model(lexical_model,lexical_data,device)
    pcs_syntactic=training_model.eval_indepence_model(syntactic_model,syntactic_data,device)

    return [pcs_layout,pcs_lexical,pcs_syntactic]

def caa_forsee(meta:caches_manager.ForseeCachesMetatadata,sp,class_number,device,prefer=None):
    model=caa_model.prepare_forsee_model(sp,device,class_number)
    forsee_paramters=torch.load(meta.preference_model)
    model.load_state_dict(forsee_paramters)
    
    model.preference_module.set_prefer(prefer)

    return model

def caa_forsee_vanille(meta:caches_manager.ForseeCachesMetatadata,sp,class_number,device):
    model=caa_model.prepare_forsee_model(sp,device,class_number)
    forsee_paramters=torch.load(meta.vanille_preference_model)
    model.load_state_dict(forsee_paramters)
    

    return model


def caa_forsee_classification(meta:caches_manager.ForseeCachesMetatadata,sp,class_number,data,device,reverse=False):
    layout_model,lexical_model,syntactic_model=caa_model.prepare_independence_model(sp,device,class_number)
    preference_model=caa_model.prepare_forsee_model(sp,device,class_number)


    layout_parameters=torch.load(meta.layout_extractor)
    lexical_parameters=torch.load(meta.lexical_extractor)
    syntactic_parameters=torch.load(meta.syntactic_extractor)
    preference_parameters=torch.load(meta.preference_model)
    
    layout_model.load_state_dict(layout_parameters)
    lexical_model.load_state_dict(lexical_parameters)
    syntactic_model.load_state_dict(syntactic_parameters)
    preference_model.load_state_dict(preference_parameters)

    layout_data=prepare_torch.ForseeLayoutDataset(data)
    lexical_data=prepare_torch.ForseeLexicalDataset(data)
    syntactic_data=prepare_torch.ForseeSyntacticDataset(data)

    iterators=[]

    iterators.append(training_model.validate_forsee(preference_model,data,device))
    iterators.append(training_model.validate_indepence_model(layout_model,layout_data,device))
    iterators.append(training_model.validate_indepence_model(lexical_model,lexical_data,device))
    iterators.append(training_model.validate_indepence_model(syntactic_model,syntactic_data,device))


    error_stat=[[] for _ in range(4)]
    error_times=[0.]*4
    for group in zip(*iterators):
        #forsee_pv,lay_pv,lex_pv,syn_pv=group
        for i,(y_hat,y) in enumerate(group):
            predict_cl_vec=torch.argmax(y_hat,1)
            ground_true_cl_vec=torch.argmax(y,1)
            
            if reverse:
                judge_vec=torch.where(predict_cl_vec==ground_true_cl_vec,torch.tensor([1],device=device),torch.tensor([0],device=device))
            else:
                judge_vec=torch.where(predict_cl_vec==ground_true_cl_vec,torch.tensor([0],device=device),torch.tensor([1],device=device))
            error_times[i]+=torch.sum(judge_vec).item()

            predict_max_v,_=torch.max(y_hat,1)
            #print(predict_max_v,judge_vec)
            error_stat[i]+=(predict_max_v*judge_vec).cpu().numpy().tolist()
    return error_stat,error_times

def caa_measure_forsee(sp,data,device,out_file=None,k=10):
    
    class_number=data.class_number
    layout_model,lexical_model,syntactic_model=caa_model.prepare_independence_model(sp,device,class_number)
    preference_model=caa_model.prepare_forsee_model(sp,device,class_number)

    status={}
    
    layout_data=prepare_torch.ForseeLayoutDataset(data)
    lexical_data=prepare_torch.ForseeLexicalDataset(data)
    syntactic_data=prepare_torch.ForseeSyntacticDataset(data)

    layout_train,layout_test=prepare_torch.fold_dataset(layout_data,k)
    lexical_train,lexical_test=prepare_torch.fold_dataset(lexical_data,k)
    syntactic_train,syntactic_test=prepare_torch.fold_dataset(syntactic_data,k)
    data_train,data_test=prepare_torch.fold_dataset(data,k)

    status['layout']=training_model.measure_model(layout_model,layout_train,layout_test,device,epoch=200)
    status['lexical']=training_model.measure_model(lexical_model,lexical_train,lexical_test,device,epoch=200)
    status['syntactic']=training_model.measure_model(syntactic_model,syntactic_train,syntactic_test,device,epoch=200)
    status['vanille_preference']=training_model.measure_model(preference_model,data_train,data_test,device,epoch=200,model_call_handle=training_model.forsee_call)
    
    if out_file:
        dataapi.pickle_save(status,out_file)
        print(f'save to {out_file}.')
    return status
    
def caa_build_raw_data(dataset_dir):
    
    pass