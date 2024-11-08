from dataclasses import dataclass
import os
import shutil
import hashlib
import torch

@dataclass
class ForseeCachesMetatadata:
    training_raw_data_file:str=""
    training_refine_data_file:str=""
    test_raw_data_file:str=""
    test_refine_data_file:str=""
    lexical_file:str=""
    syntactic_file:str=""
    training_default_task_data_file:str=""
    test_default_task_data_file:str=""

    tokens_file:str=""
    asts_file:str=""


    layout_extractor:str=""
    lexical_extractor:str=""
    syntactic_extractor:str=""

    preference_model:str=""
    vanille_preference_model:str=""
    DLCAIS:str=""
    PbNN:str=""

    caches_dir:str=""


    

    @staticmethod
    def auto(dataset_dir,store_loc=None,flag="dataset"):
        
        if store_loc is None:
            store_loc='caches'
        
        if not os.path.exists(store_loc):
            os.mkdir(store_loc)
        
        caches_dir=f'{store_loc}/{flag}_{hashlib.md5(dataset_dir.encode()).hexdigest()}'
        
        if not os.path.exists(caches_dir):
            os.mkdir(caches_dir)
        
            
        obj=ForseeCachesMetatadata()

        obj.caches_dir=caches_dir
        

        obj.training_raw_data_file=os.path.join(caches_dir,'data.pt')
        obj.training_refine_data_file=os.path.join(caches_dir,'refine_data.pt')
        obj.lexical_file=os.path.join(caches_dir,'lexical.pt')
        obj.syntactic_file=os.path.join(caches_dir,'syntactic.pt')
        obj.training_default_task_data_file=os.path.join(caches_dir,'to_data.pt')

        obj.test_refine_data_file=os.path.join(caches_dir,'test_refine_data.pt')
        obj.test_default_task_data_file=os.path.join(caches_dir,'test_to_data.pt')
        obj.test_raw_data_file=os.path.join(caches_dir,'test_data.pt')
        
        obj.layout_extractor=os.path.join(caches_dir,'layout_extractor.pt')
        obj.lexical_extractor=os.path.join(caches_dir,'lexical_extractor.pt')
        obj.syntactic_extractor=os.path.join(caches_dir,'syntactic_extractor.pt')
        
        obj.tokens_file=os.path.join(caches_dir,'tokens.pt')
        obj.asts_file=os.path.join(caches_dir,'ast_dump.pt')


        obj.preference_model=os.path.join(caches_dir,'preference_model.pt')
        obj.vanille_preference_model=os.path.join(caches_dir,'vannile_preference_modle.pt')
        obj.DLCAIS=os.path.join(caches_dir,'DLCAIS.pt')
        obj.PbNN=os.path.join(caches_dir,'PbNN.pt')

        return obj
    
    def partial_model(self,use_layout,use_lexical,use_syntactic):
        return os.path.join(self.caches_dir,f'partial_model_{use_layout}_{use_lexical}_{use_syntactic}.pt')

    @staticmethod
    def external_caches(external_caches_dir,group=None):
        
        if group is None: group=""
        
        raw_features=os.path.join(external_caches_dir,f'raw_data{group}.pt')
        refine_features=os.path.join(external_caches_dir,f'refine_data{group}.pt')
        torch_data=os.path.join(external_caches_dir,f'task_orient{group}.pt')
            
        return raw_features,refine_features,torch_data

    
# def make_dataset(dataset_dir,lang,rebuild):
    
#     metadata=ForseeCachesMetatadata.auto(dataset_dir)
#     dataset=None
#     refine_dataset=None


#     if rebuild:
#         caches=os.listdir(dataset_dir)

#         for e in caches:
#             os.remove(os.path.join(dataset_dir,e))


#     if os.path.exists(metadata.training_raw_data_file):
#         dataset = torch.load(metadata.training_raw_data_file)

#     else:
#         dataset=preprocessor.build_raw_dataset(dataset_dir,lang,preprocessor.list_author_in_dir,metadata.training_raw_data_file)
        

#     if os.path.exists(metadata.training_refine_data_file):
#         refine_dataset=torch.load(metadata.training_refine_data_file)

#     else:
#         lexcical_module=preprocessor.build_lexical_preprocess_component(dataset,metadata.lexical_file)
#         syntactic_module=preprocessor.build_syntactic_preprocess_component(dataset,metadata.syntactic_file)
#         refine_dataset=preprocessor.build_refine_dataset(dataset,lexcical_module,syntactic_module,metadata.training_refine_data_file)

#     return dataset,refine_dataset

# def load_dataset_auto(dataset_dir):
#     metadata=ForseeCachesMetatadata.auto(dataset_dir)
    
#     if os.path.exists(metadata.training_refine_data_file):
#         return torch.load(metadata.training_refine_data_file)
    
#     return None
    
    #extract_gcj_cpp(r'D:\Code\doctor\code-semantic-classcification\data\pack\gcj_cpp',r'D:\Code\doctor\code-semantic-classcification\data\gcj_cpp')