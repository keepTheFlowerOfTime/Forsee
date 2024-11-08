from __future__ import annotations
# CAA: Code Authorship Attribution
from dataclasses import dataclass
from codesecurity.feature import CommonFeatureSet,TfidfModule
from codesecurity.feature.api import read_bytes,create_ast_obj
from codesecurity.data.api import list_dataset,GroupPipe,pickle_load,list_all_file_in_dir
from codesecurity.data.parallel import do_parallel

import pickle
import numpy as np
import os

from codesecurity.feature.objects import Ast, AstEdge
from codesecurity.tasks.code_authorship_attribution.caches_manager import ForseeCachesMetatadata


def sample_is_good(common_feature:CommonFeatureSet):
    if common_feature is None: return False
    # token number > 20
    if len(common_feature.tokens)<=20: return False
    # ast node > 10
    if len(common_feature.nodes)<=10: return False
    
    return True

def type_pair(ast:Ast):
    edges=ast.edges
    return [ast.ast_type(edge).id for edge in edges]

class CAAFeatureBuilder:
    def __init__(self,dataset_dir,label2id=None,list_author_handle=None,minimum_number=1,maximun_number=2**32-1) -> None:
        if label2id is None:
            label2id={}


        self.minimun_number=minimum_number
        self.maximin_number=maximun_number
        self.dataset_dir=dataset_dir
        self.list_author_handle=list_author_handle
        self.label2id=label2id

    @staticmethod
    def feature_delegate(sample_path,lang=None):
        common_feature=CommonFeatureSet.from_file(sample_path,lang)
        if not sample_is_good(common_feature):
            return None,sample_path
        
        return common_feature,sample_path

    def iter_group(self):
        sample_pairs=list_dataset(self.dataset_dir,self.list_author_handle)
        
        #print(list(sample_pairs))
        
        for sample_paths,sample_label in sample_pairs:
            counter=0    
            
            pairs=[(CAAFeatureBuilder.feature_delegate,[sample_path]) for sample_path in sample_paths]
            pairs=pairs[:min(len(pairs),self.maximin_number)]
            
            group_labels=[]
            group_codewaves=[]
            group_tokens=[]
            group_origin_paths=[]
            group_ast=[]
            
            common_features=do_parallel(pairs)
            
            for common_feature,sample_path in common_features:
                if common_feature is None: continue
                group_labels.append(self.get_id(sample_label))
                group_codewaves.append(CAAFeatureBuilder.get_codewave(sample_path))
                group_tokens.append(common_feature.tokens)
                group_origin_paths.append(sample_path)
                group_ast.append(common_feature.ast_object)
                counter+=1
            
            if counter<self.minimun_number:
                print(f"number of sample for {sample_label} is {counter}, not enough for prediction.")
                continue

            print(f"good sample for {sample_label}: {counter}/{len(sample_paths)}")
    
            yield group_tokens,group_ast,group_codewaves,group_origin_paths,group_labels

    def get_id(self,label):       
        if label not in self.label2id:
            self.label2id[label]=len(self.label2id)
        
        return self.label2id[label]
    
    @staticmethod
    def get_codewave(sample_path,dev=1):
        result=[]
        with open(sample_path,'rb') as f:
            for line in f:
                line_number=len(line)+dev
                result.append(line_number)
                
        return result
    
    def save(self,out_file):
        with open(out_file,'wb') as f:
            pickle.dump(self,f)


    @staticmethod
    def _build(pairs,label):
        group_labels=[]
        group_codewaves=[]
        group_tokens=[]
        group_origin_paths=[]
        group_ast=[]
        
        common_features=do_parallel(pairs)
        
        for common_feature,sample_path in common_features:
            if common_feature is None: continue
            group_labels.append(label)
            group_codewaves.append(CAAFeatureBuilder.get_codewave(sample_path))
            group_tokens.append(common_feature.tokens)
            group_origin_paths.append(sample_path)
            group_ast.append(common_feature.ast_object)
        
        return group_tokens,group_ast,group_codewaves,group_origin_paths,group_labels

    @staticmethod
    def build(repo_dir,label,lang=None,min_number=1,max_number=2**32-1):
        sample_paths=list_all_file_in_dir(repo_dir)
        try:
            pairs=[(CAAFeatureBuilder.feature_delegate,[sample_path,lang]) for sample_path in sample_paths]
            pairs=pairs[:min(len(pairs),max_number)]
            
            result=CAAFeatureBuilder._build(pairs,label)
            if len(result[0])<min_number:
                print(f"number of sample for {label} is {len(result[0])}, not enough for prediction.")
                return None
            
            print(f"good sample for {repo_dir}: {len(result[0])}/{len(sample_paths)}")
            
            return result
        
        except Exception as e:
            print(f"analyze repo for {label} is failed, {e}.")
            return None 
        
        

        

            
@dataclass
class ForseeSuperParameter:
    lexical_vector_dim:int=0
    syntactic_vector_dim:int=0
    layout_vector_dim:int=0

    lex_hidden_dim:int=0
    lex_channel:int=0

    syn_hidden_dim:int=0
    syn_channel:int=0

    lay_hidden_dim:int=0
    lay_channel:int=0
    
    lay_max_value:int=256
    
    batch_size:int=128
    lr:float=0.001


    @staticmethod
    def instance(name):
        if name=='default':
            return ForseeSuperParameter.default()
        elif name=='gcj_cpp' or name=='gcj':
            return ForseeSuperParameter.gcj()
        elif name=='github_java' or name=='java40':
            return ForseeSuperParameter.github_java()
        elif name=='github_c':
            return ForseeSuperParameter.github_c()
        elif name=="persp_cpp" or name=='perspective_cpp':
            return ForseeSuperParameter.perspective_cpp()
        else:
            return ForseeSuperParameter.default()
        
    @staticmethod
    def default():
        obj=ForseeSuperParameter()
        obj.lexical_vector_dim=4500
        obj.syntactic_vector_dim=4500
        obj.layout_vector_dim=256

        obj.lex_channel=32
        obj.lex_hidden_dim=512
        obj.syn_channel=32
        obj.syn_hidden_dim=512
        obj.lay_channel=32
        obj.lay_hidden_dim=512

        return obj
    
    

    @staticmethod
    def github_c():
        obj=ForseeSuperParameter()
        obj.lexical_vector_dim=2500
        obj.syntactic_vector_dim=4500
        obj.layout_vector_dim=256

        obj.lex_channel=16
        obj.lex_hidden_dim=256
        obj.syn_channel=16
        obj.syn_hidden_dim=256
        obj.lay_channel=16
        obj.lay_hidden_dim=256

        return obj

    @staticmethod
    def github_java():
        obj=ForseeSuperParameter()
        obj.lexical_vector_dim=2500
        obj.syntactic_vector_dim=2500
        obj.layout_vector_dim=256

        obj.lex_channel=16
        obj.lex_hidden_dim=256
        obj.syn_channel=16
        obj.syn_hidden_dim=256
        obj.lay_channel=16
        obj.lay_hidden_dim=256

        return obj
    
    @staticmethod
    def perspective_cpp():
        obj=ForseeSuperParameter()
        obj.lexical_vector_dim=4500
        obj.syntactic_vector_dim=2500
        obj.layout_vector_dim=256

        obj.lex_channel=16
        obj.lex_hidden_dim=128
        obj.syn_channel=16
        obj.syn_hidden_dim=128
        obj.lay_channel=16
        obj.lay_hidden_dim=256

        return obj

    @staticmethod
    def gcj():
        obj=ForseeSuperParameter()
        obj.lexical_vector_dim=2500
        obj.syntactic_vector_dim=4500
        obj.layout_vector_dim=256

        obj.lex_channel=16
        obj.lex_hidden_dim=256
        obj.syn_channel=16
        obj.syn_hidden_dim=256
        obj.lay_channel=16
        obj.lay_hidden_dim=256

        return obj

@dataclass
class DLCAISSuperParameter:
    hidden_dim:int=0
    input_dim:int=0
    
    batch_size:int=128
    epoch_number:int=100
    
    @staticmethod
    def default():
        obj=DLCAISSuperParameter()
        obj.input_dim=2500
        obj.hidden_dim=256
        return obj

class ForseeFeatureBuilder:
    def __init__(self,sp,lexical_tfidf:TfidfModule,syntactic_tfidf:TfidfModule):
        self.sp=sp
        self.lexical_tfidf=lexical_tfidf
        self.syntactic_tfidf=syntactic_tfidf


    def build(self,group_tokens,group_ast,group_codewaves,group_origin_paths,group_labels):
        group_ast=self.syntactic_feature(group_ast)
        group_tokens=self.lexical_feature(group_tokens)
        group_codewaves=self.layout_feature(group_codewaves)
        feature_objs=[]
        for i in range(len(group_labels)):
            feature_obj=ForseeFeature(
                lexical=group_tokens[i],
                syntactic=group_ast[i],
                layout=group_codewaves[i],
                origin_path=group_origin_paths[i],
                label=group_labels[i]
            )
            feature_objs.append(feature_obj)

        return feature_objs
    
    def layout_feature(self,group_codewaves):
    
        layout=[None]*len(group_codewaves)
        
        for i in range(len(layout)):
            codewave=group_codewaves[i]
            v=np.array(codewave)
            v=np.clip(v,0,self.sp.lay_max_value)

            if len(v)>self.sp.layout_vector_dim:
                v=v[:self.sp.layout_vector_dim]

            elif len(v)<self.sp.layout_vector_dim:
                v=np.pad(v,[0,self.sp.layout_vector_dim-len(v)])

            layout[i]=v
        return layout
     
    def lexical_feature(self,group_tokens):
        lexical_tfidf=self.lexical_tfidf

        lexical_docs=[]
        for index in range(len(group_tokens)):
            tokens=group_tokens[index]
            lexical_docs.append(tokens)

        return lexical_tfidf.get_tfidf_vec(lexical_docs,self.sp.lexical_vector_dim)
             
    def syntactic_feature(self,group_syntactic):
        syntactic_tfidf=self.syntactic_tfidf

        syntactic_docs=[]
        for index in range(len(group_syntactic)):
            edges=group_syntactic[index].edges
            syntactic_docs.append([group_syntactic[index].ast_type(e).id for e in edges])

            
        return syntactic_tfidf.get_tfidf_vec(syntactic_docs,self.sp.syntactic_vector_dim)


    @staticmethod
    def build_from_pipe(sp,meta:ForseeCachesMetatadata=None,lexical_file=None,syntactic_file=None,raw_data=None):

        if meta:
            lexical_file=meta.lexical_file
            syntactic_file=meta.syntactic_file
            raw_data=meta.training_raw_data_file

        lexical_module=TfidfModule.load(lexical_file)
        syntactic_module=TfidfModule.load(syntactic_file)
        batch_group:GroupPipe=pickle_load(raw_data)

        forsee_feature_builder=ForseeFeatureBuilder(sp,lexical_module,syntactic_module)
        samples=[]
        for batch in batch_group.iter_group():
            for group in batch:
                samples+=forsee_feature_builder.build(*group)

        return ForseeFeatures(samples,batch_group.addon)
    
class ForseeFeature:
    def __init__(self,origin_path,layout,lexical,syntactic,label) -> None:
        self.origin_paths=origin_path
        self.layout=layout
        self.lexical=lexical
        self.syntactic=syntactic
        self.label=label

class ForseeFeatures:
    def __init__(self,samples:list[ForseeFeature],id_mapping) -> None:
        self.samples=samples
        self.id_mapping=id_mapping

    def save(self,out_file):
        if out_file:
            with open(out_file,'wb') as f:
                pickle.dump(self,f)


def build_lexical_tfidfmodule(raw_features:CAAFeatureBuilder,caches_file=None):

    if os.path.exists(caches_file):
        return TfidfModule.load(caches_file)

    module=TfidfModule()

    docs=raw_features.tokens
    module.add_documents(docs)

    print(f"add documents {len(docs)} for tfidf module.")

    module.build()
    
    if caches_file:
        module.save(caches_file)
    
    return module

def build_syntactic_tfidfmodule(raw_features:CAAFeatureBuilder,caches_file=None):
    def type_pair(ast:Ast):
        edges=ast.edges
        return [ast.ast_type(edge).id for edge in edges]
    
    if os.path.exists(caches_file):
        return TfidfModule.load(caches_file)

    module=TfidfModule()
    
    docs=[type_pair(e) for e in raw_features.ast]
    module.add_documents(docs)
    module.build()
    
    if caches_file:
        module.save(caches_file)
    
    return module

def prepare_forsee_features(dataset_dir,meta:ForseeCachesMetatadata,sp:ForseeSuperParameter,list_author_handle=None,id_mapping=None,min_number=1,max_number=2**32-1,lexical_file=None,syntactic_file=None,raw_data=None):
    
    if meta:
        lexical_file=meta.lexical_file
        syntactic_file=meta.syntactic_file
        raw_data=meta.training_raw_data_file

    caa_feature_builder=CAAFeatureBuilder(dataset_dir,id_mapping,list_author_handle=list_author_handle,minimum_number=min_number,maximun_number=max_number)
    group_pipe=GroupPipe(raw_data,addon=caa_feature_builder.label2id)

    lexical_module=TfidfModule()
    syntactic_module=TfidfModule()
    for group in caa_feature_builder.iter_group():
        group_tokens,group_ast,group_codewaves,group_origin_paths,group_labels=group
        lexical_module.add_documents(group_tokens)
        syntactic_module.add_documents([type_pair(e) for e in group_ast])
        group_pipe.add_group(group)

    group_pipe.save(caa_feature_builder.label2id)
    lexical_module.build()
    syntactic_module.build()

    lexical_module.save(lexical_file)
    syntactic_module.save(syntactic_file)

    return ForseeFeatureBuilder.build_from_pipe(sp,meta,lexical_file=lexical_file,syntactic_file=syntactic_file,raw_data=raw_data)

def prepare_forsee_features_for_external_data(dataset_dir,sp:ForseeSuperParameter,id_mapping,lexical_file,syntactic_file,external_raw_file,list_author_handle=None,min_number=1,max_number=2**32-1):

    caa_feature_builder=CAAFeatureBuilder(dataset_dir,id_mapping,list_author_handle=list_author_handle,minimum_number=min_number,maximun_number=max_number)

    external_group_pipe=GroupPipe(external_raw_file)

    for group in caa_feature_builder.iter_group():
        external_group_pipe.add_group(group)

    external_group_pipe.save(caa_feature_builder.label2id)


    return ForseeFeatureBuilder.build_from_pipe(sp,lexical_file=lexical_file,syntactic_file=syntactic_file,raw_data=external_raw_file)

def build_forsee_features_for_single_repo(repo_dirs,labels,sp:ForseeSuperParameter,id_mapping,lexical_file,syntactic_file,lang=None,min_number=1,max_number=2**32-1):
    
    lexical_module=pickle_load(lexical_file)
    syntactic_module=pickle_load(syntactic_file)
    builder=ForseeFeatureBuilder(sp,lexical_module,syntactic_module)
    
    features=[]
    
    for repo_dir,label in zip(repo_dirs,labels):
        group=CAAFeatureBuilder.build(repo_dir,label,lang,min_number,max_number)
        if group:
            features+=builder.build(*group)
    return ForseeFeatures(features,id_mapping)
# def prepare_forsee_features(raw_features:CAARawFeatures,sp:ForseeSuperParameter,lexical_tfidf:TfidfModule,syntactic_tfidf:TfidfModule):

#     forsee_features=ForseeFeatures(raw_features,lexical_tfidf,syntactic_tfidf,sp)
    
#     return forsee_features,lexical_tfidf,syntactic_tfidf



    # def get_graph_nodes(self,max_real_node,max_virtual_node):
    #     real_nodes=self.get_single_nodes()
    #     virtual_nodes=self.get_complex_nodes()
        
    #     if len(real_nodes)<max_real_node:
    #         real_nodes+=[None]*(max_real_node-len(real_nodes))
    #     elif len(real_nodes)>max_real_node:
    #         real_nodes=real_nodes[:max_real_node]
            
    #     if len(virtual_nodes)<max_virtual_node:
    #         virtual_nodes+=[None]*(max_virtual_node-len(virtual_nodes))
    #     elif len(virtual_nodes)>max_virtual_node:
    #         virtual_nodes=virtual_nodes[:max_virtual_node]
        
    #     real_node_tokens=list(map(lambda x:x.text if x else "",real_nodes))
    #     virtual_node_tokens=list(map(lambda x:x.type if x else "",virtual_nodes))
            
    #     return real_node_tokens+virtual_node_tokens
    
    # def get_graph_edge_ast(self,max_real_node,max_virtual_node):
        
    #     origin2mapping=self.get_index_mapping(max_real_node,max_virtual_node)
        
    #     result=[]
    #     for edge in self.ast_edges:
    #         start_node,end_node=edge
    #         start_index,end_index=origin2mapping.get(start_node.id),origin2mapping.get(end_node.id)
            
    #         if start_index and end_index:
    #             result.append((start_index,end_index))
                
    #     return result
    
    # def get_graph_edge_ncs(self,max_real_node,max_virtual_node):
    #     real_node_number=len(self.get_single_nodes())
    #     real_node_number=min(max_real_node,real_node_number)
        
    #     result=[]
    #     for i in range(real_node_number-1):
    #         result.append((i,i+1))
        
    #     return result
    
    # def get_single_node_types(self):
    #     nodes=self.get_single_nodes()

    #     return [node.type for node in nodes]

    # def get_complex_node_types(self):
    #     nodes=self.get_complex_nodes()

    #     return [node.type for node in nodes]

    # def get_index_mapping(self,max_real_node,max_virtual_node):
    #     origin_mapping={}
    #     for node_id in self.node_properties:
    #         node_property:NodeProperty=self.node_properties[node_id]
    #         if node_property.is_leave:
    #             if node_property.mapping_index<max_real_node:
    #                 origin_mapping[node_id]=node_property.mapping_index
                
    #         else:
    #             if node_property.mapping_index<max_virtual_node:
    #                 origin_mapping[node_id]=max_real_node+node_property.mapping_index
        
    #     return origin_mapping


    