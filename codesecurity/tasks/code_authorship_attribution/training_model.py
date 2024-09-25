import torch
import os
import torch.utils.data as torchdata

from codesecurity.tasks.code_authorship_attribution.caches_manager import ForseeCachesMetatadata
from codesecurity.tasks.code_authorship_attribution.preprocessing import DLCAISSuperParameter, ForseeSuperParameter
from codesecurity.tasks.code_authorship_attribution.model import prepare_DLCAIS, prepare_forsee_model,prepare_independence_model,IndependenceModel
from codesecurity.tasks.code_authorship_attribution.prepare_torch import ForseeLayoutDataset,ForseeLexicalDataset,ForseeSyntacticDataset,ForseePartialDataset,layout_enhance,lexical_enhance,syntactic_enhance,combine_enhance



import codesecurity.tasks.code_authorship_attribution.model as caa_model
import codesecurity.nn.api as nn_api

import sklearn.ensemble as sken
import numpy as np

from collections.abc import Iterable

def train_model(model:torch.nn.Module,training_data:torchdata.DataLoader,test_data:torchdata.DataLoader,device,loss_func=torch.nn.CrossEntropyLoss(),model_call_handle=None,out_file=None,epoch_number=150,lr=1e-3,weight_decay=0,reinforce=None):

    if model_call_handle is None:
        model_call_handle=lambda model,x:model(x[0])

    highest_accuracy=eval_model(model,test_data,device,loss_func,model_call_handle)

    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)

    for i in range(epoch_number):
        total_number=0
        print(f"epoch : {i}:")
        model.train()

        #total_test_loss=0
        total_accuracy=0

        for e in training_data:
            y=e[-1]
            x=e[:-1]
            if reinforce is not None:
                x=reinforce(x)
            y_hat=model_call_handle(model,e[:-1])
            #y_hat=torch.softmax(y_hat,1)
            if len(y.size())==1:
                y=torch.eye(y_hat.size(-1))[y]
            y=y.to(device)
            loss= loss_func(y_hat,y.double())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():

                class_number=y_hat.size(-1)

                _,maxk=torch.topk(y_hat,min(class_number,5),dim=-1)


                total_accuracy+=(y.argmax(1).view(-1,1) == maxk[:,0:1]).sum().item()
                total_number+=y.size(0)
        print(f"train top1: {total_accuracy/total_number}")

        #model.eval()
        now_accuary=eval_model(model,test_data,device,loss_func,model_call_handle)

        if out_file:
            if now_accuary>=highest_accuracy:
                torch.save(model.state_dict(),out_file)
                now_accuary=highest_accuracy

def validate_model(model:torch.nn.Module,data:torchdata.DataLoader,device,model_call_handle=None):
    if model_call_handle is None:
        model_call_handle=lambda model,x:model(x[0])
    model.eval()
    with torch.no_grad():
        for e in data:
            y=e[-1]
            y=y.to(device)
            y_hat=model_call_handle(model,e[:-1])

            y_hat=torch.softmax(y_hat,1)
            if len(y.size())==1:
                y=torch.eye(y_hat.size(-1),device=device)[y]

            yield y_hat,y
            
def eval_model(model:torch.nn.Module,data:torchdata.DataLoader,device,loss_func=torch.nn.CrossEntropyLoss(),model_call_handle=None):

    if model_call_handle is None:
        model_call_handle=lambda model,x:model(*x)

    total_number=0
    total_test_loss=0
    total_accuracy=0
    total_acc5=0
    number_of_iter=0
    model.eval()
    with torch.no_grad():
        for e in data:
            y=e[-1]
            y=y.to(device)
            y_hat=model_call_handle(model,e[:-1])

            if len(y.size())==1:
                y=torch.eye(y_hat.size(-1),device=device)[y]
            #print(y.size(),y_hat.size())

            #y_hat=torch.softmax(y_hat,1)

            loss= loss_func(y_hat,y.double())
            total_test_loss = total_test_loss + loss.item()

            class_number=y_hat.size(-1)

            _,maxk=torch.topk(y_hat,min(5,class_number),dim=-1)



            total_accuracy+=(y.argmax(1).view(-1,1) == maxk[:,0:1]).sum().item()
            total_acc5+=(y.argmax(1).view(-1,1)== maxk).sum().item()
            # y_hat_group=torch.stack([y_hat[i*test_dataset.group:(i+1)*test_dataset.group,:] for i in range(1)])
            # y_hat_group_eval=(torch.max(y_hat_group,2,keepdim=True)[0]==y_hat_group)

            # y_hat_group_eval=torch.sum(y_hat_group_eval,1)
            # y_group_eval=torch.stack([y[i*test_dataset.group] for i in range(1)])

            # group_acc=(y_hat_group_eval.argmax(1) == y_group_eval.argmax(1)).sum()
            # total_group_acc+=group_acc
            number_of_iter+=1
            total_number+=y.size(0)
        print(f'loss :{total_test_loss/number_of_iter} top1:{total_accuracy/total_number} top5: {total_acc5/total_number}')
    
    return total_accuracy/total_number

def forsee_call(model,x):
    if isinstance(x,Iterable):
        return model(*x)
    else:
        return model(x)

def train_forsee(training_data,test_data,class_number,meta:ForseeCachesMetatadata,sp:ForseeSuperParameter,device=torch.device('cpu'),use_caches=True):
    

    layout_model,lexical_model,syntactic_model=prepare_independence_model(sp,device,class_number)


    if use_caches and os.path.exists(meta.layout_extractor):
        layout_model.load_state_dict(torch.load(meta.layout_extractor))
    layout_training_data=ForseeLayoutDataset(training_data)
    layout_test_data=ForseeLayoutDataset(test_data)
    train_independence_model(layout_model,layout_training_data,layout_test_data,device,meta.layout_extractor,enhance=layout_enhance,epoch=200)
        
    if use_caches and os.path.exists(meta.lexical_extractor):
        lexical_model.load_state_dict(torch.load(meta.lexical_extractor))
    lexical_training_data=ForseeLexicalDataset(training_data)
    lexical_test_data=ForseeLexicalDataset(test_data)
    train_independence_model(lexical_model,lexical_training_data,lexical_test_data,device,meta.lexical_extractor,enhance=lexical_enhance)
        
    if use_caches and os.path.exists(meta.syntactic_extractor):
        syntactic_model.load_state_dict(torch.load(meta.syntactic_extractor))

    syntactic_training_data=ForseeSyntacticDataset(training_data)
    syntactic_test_data=ForseeSyntacticDataset(test_data)
    train_independence_model(syntactic_model,syntactic_training_data,syntactic_test_data,device,meta.syntactic_extractor,enhance=syntactic_enhance)


    preference_network=prepare_forsee_model(sp,device,class_number)
    
    
    embeding_module=preference_network.embeding_module
    preference_module=preference_network.preference_module

    #embeding_module.layout_extractor.named_parameters()
    embeding_module.layout_extractor.load_state_dict(layout_model.extractor.state_dict())
    embeding_module.lexical_extractor.load_state_dict(lexical_model.extractor.state_dict())
    embeding_module.syntactic_extractor.load_state_dict(syntactic_model.extractor.state_dict())

    embeding_module.requires_grad_(False)

    preference_module.layout_w.requires_grad=False
    preference_module.syntactic_w.requires_grad=False
    preference_module.lexical_w.requires_grad=False
    
    training_data=torchdata.DataLoader(training_data,batch_size=sp.batch_size)
    test_data=torchdata.DataLoader(test_data,batch_size=sp.batch_size)

    train_model(preference_network,training_data,test_data,device,model_call_handle=forsee_call,out_file=meta.preference_model,reinforce=combine_enhance)
    
    return preference_network,[layout_model,lexical_model,syntactic_model]
    

def caa_patial_forsee_training(training_data,test_data,meta:ForseeCachesMetatadata,sp,class_number,device,use_layout,use_lexical,use_syntactic):
    preference_model=caa_model.prepare_partial_forsee_model(sp,device,class_number,use_layout,use_lexical,use_syntactic)
    independent_models=caa_model.prepare_select_independence_models(sp,device,class_number,use_layout,use_lexical,use_syntactic)
    parameter_paths=[]
    if use_layout:
        parameter_paths.append(meta.layout_extractor)
    if use_lexical:
        parameter_paths.append(meta.lexical_extractor)
    if use_syntactic:
        parameter_paths.append(meta.syntactic_extractor)
    
    for i in range(len(parameter_paths)):
        independent_models[i].load_state_dict(torch.load(parameter_paths[i]))
        preference_model.extractors[i].load_state_dict(independent_models[i].extractor.state_dict())

    preference_model.frozen_no_classicifer_parameters()

    training_data=ForseePartialDataset(training_data,use_layout,use_lexical,use_syntactic)
    test_data=ForseePartialDataset(test_data,use_layout,use_lexical,use_syntactic)

    training_data=torchdata.DataLoader(training_data,batch_size=sp.batch_size)
    test_data=torchdata.DataLoader(test_data,batch_size=sp.batch_size)

    train_model(preference_model,training_data,test_data,device,model_call_handle=forsee_call,out_file=meta.partial_model(use_layout,use_lexical,use_syntactic))

    return preference_model

def caa_vanille_forsee_training(training_data,test_data,meta:ForseeCachesMetatadata,sp,class_number,device):
    preference_model=caa_model.prepare_forsee_model(sp,device,class_number)

    training_data=ForseePartialDataset(training_data,True,True,True)
    test_data=ForseePartialDataset(test_data,True,True,True)

    training_data=torchdata.DataLoader(training_data,batch_size=sp.batch_size)
    test_data=torchdata.DataLoader(test_data,batch_size=sp.batch_size)

    train_model(preference_model,training_data,test_data,device,model_call_handle=forsee_call,out_file=meta.vanille_preference_model)

    return preference_model

def train_independence_model(model,training_data,test_data,device=torch.device('cpu'),out_file=None,enhance=None,epoch=150):    
    training_data=torchdata.DataLoader(training_data,batch_size=64)
    test_data=torchdata.DataLoader(test_data,batch_size=64)

    train_model(model,training_data,test_data,device,out_file=out_file,reinforce=enhance,epoch_number=epoch)

    return model

def measure_model(model,training_data,test_data,device=torch.device('cpu'),enhance=None,epoch=150,model_call_handle=None):    
    
    training_data=torchdata.DataLoader(training_data,batch_size=64)
    test_data=torchdata.DataLoader(test_data,batch_size=64)

    status=nn_api.measure_train_model(model,training_data,test_data,device,reinforce=enhance,epoch_number=epoch,model_call_handle=model_call_handle)

    return status

def train_DLCAIS_nn(training_data,test_data,class_number,meta:ForseeCachesMetatadata,sp:DLCAISSuperParameter,device=torch.device('cpu'),use_caches=True):
    model=prepare_DLCAIS(sp,device,class_number)
    if use_caches and os.path.exists(meta.DLCAIS):
        model.load_state_dict(torch.load(meta.DLCAIS))
        return model
    training_data=torchdata.DataLoader(training_data,batch_size=sp.batch_size)
    test_data=torchdata.DataLoader(test_data,batch_size=sp.batch_size)
    
    train_model(model,training_data,test_data,device,out_file=meta.DLCAIS,epoch_number=1000,lr=1e-4,weight_decay=0.00001)
    return model

def train_DLCAIS_RFC(training_data,test_data,class_number,meta:ForseeCachesMetatadata,sp:DLCAISSuperParameter,model=None,device=torch.device('cpu')):
    classifier=sken.RandomForestClassifier(n_estimators=300)

    if model is None:
        model=prepare_DLCAIS(sp,device,class_number)
        if not os.path.exists(meta.DLCAIS): raise Exception("DLCAIS model not found")
    
        model.load_state_dict(torch.load(meta.DLCAIS))

    training_inputs=[]
    for e in embeding_model(model,training_data):
        training_inputs.append(e)
        
    test_inputs=[]
    for e in embeding_model(model,test_data):
        test_inputs.append(e)
        
    training_embeding=np.concatenate([e[0] for e in training_inputs],axis=0)
    test_embeding=np.concatenate([e[0] for e in test_inputs],axis=0)
    
    traning_labels=np.concatenate([e[1].numpy() for e in training_inputs],axis=0)
    test_labels=np.concatenate([e[1].numpy() for e in test_inputs],axis=0)
    
    classifier.fit(training_embeding,traning_labels)
    
    pred=classifier.score(test_embeding,test_labels)
    
    print(f'RFC pred in test_set: {pred}')
    
    return classifier,test_embeding,test_labels
    

def eval_forsee(model,eval_data,device):
    
    eval_data=torchdata.DataLoader(eval_data,batch_size=64)
    
    return eval_model(model,eval_data,device,model_call_handle=forsee_call)
    
def embeding_model(model:caa_model.PreferenceNetwork,eval_data):
    eval_data=torchdata.DataLoader(eval_data,batch_size=64,shuffle=False)
    with torch.no_grad():
        for e in eval_data:
            yield model.embeding(*e[:-1]).cpu().numpy(),e[-1]

def eval_indepence_model(model,eval_data,device):
    eval_data=torchdata.DataLoader(eval_data,batch_size=64)
    
    return eval_model(model,eval_data,device)

def validate_forsee(model,data,device):
    data=torchdata.DataLoader(data,batch_size=64)

    return validate_model(model,data,device,model_call_handle=forsee_call)

def validate_indepence_model(model,data,device):
    data=torchdata.DataLoader(data,batch_size=64)

    return validate_model(model,data,device)
