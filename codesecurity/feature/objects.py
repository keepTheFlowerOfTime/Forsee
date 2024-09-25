from __future__ import annotations

from dataclasses import dataclass, field

import codesecurity.feature.api as feature_api
from codesecurity.data.objects import ProgramLanguage


@dataclass
class AstNode:
    is_leave:bool=False
    ast_type:str=""
    ast_value:str=""
    child_number:int=0

    @property
    def text(self):
        return f'{self.ast_type}({self.ast_value})'
    
    def __str__(self):
        return self.text
    
    def __hash__(self) -> int:
        return hash(id(self))
        
@dataclass
class AstEdge:
    start_index:int=0
    end_index:int=0
    
@dataclass
class Ast:
    nodes:list[AstNode]=field(default_factory=list)
    edges:list[AstEdge]=field(default_factory=list)
    
    def ast_type(self,edge:AstEdge):
        edge_type=EdgeType(self.nodes[edge.start_index].ast_type,self.nodes[edge.end_index].ast_type)    
        return edge_type
    
    def extract_ncs_flow(self):
        # 获取ncs边
        # 获取所有的AST Node，按照先序深度优先遍历，然后连接各个节点
        # 然后添加节点到子节点的边
        node_list = self.nodes
        ncs_edge = []
        # 先序深度优先遍历感觉像。
        # 所以NCS就是各个节点顺序连接。
        pre_node_range = range(len(node_list) - 1)
        next_node_range = range(1, len(node_list))
        ncs_edge = list(zip(pre_node_range, next_node_range))

        return ncs_edge
    
    def extract_lexical_common_flow(self,include_syntactic_info:bool=False):
        pass
    
    
    def get_nearby_matrix(self):
        nearby_dict={}
        nearby_dict_reverse={}
        for edge in self.edges:
            start_index,end_index=edge.start_index,edge.end_index
            if start_index not in nearby_dict:
                nearby_dict[start_index]=[]
            
            nearby_dict[start_index].append(end_index)    
            
            if end_index not in nearby_dict_reverse:
                nearby_dict_reverse[end_index]=[]
                
            nearby_dict_reverse[end_index].append(start_index)
            
        return nearby_dict,nearby_dict_reverse
    
    def search_nodes(self,node_types,node_value=None):
        
        if isinstance(node_types,str):
            node_types=[node_types]
        
        ret=[]
        for node in self.nodes:            
            if node.ast_type in node_types:
                if node_value is None or node.ast_value==node_value:
                    ret.append(node)
        return ret
    
    # def get_undirect_matrix(self):
    #     nearby_dict={}
    #     for edge in self.edges:
    #         start_index,end_index=edge
    #         start_node,end_node=self.nodes[start_index],self.nodes[end_index]
    #         if start_node not in nearby_dict:
    #             nearby_dict[start_node]=[]
            
    #         nearby_dict[start_node].append(end_index)    
            
    #         if end_node not in nearby_dict:
    #             nearby_dict[end_node]=[]
                
    #         nearby_dict[end_node].append(start_index)
            
    #     return nearby_dict
    
@dataclass
class EdgeType:
    start_type:str=""
    end_type:str=""

    @property
    def id(self):
        return f'{self.start_type}=>{self.end_type}'

class AstDirection:
    start=0
    Up=1
    Down=2    

@dataclass
class AstPath:
    nodes:list[AstNode]=field(default_factory=list)
    directions:list[int]=field(default_factory=list)

    def extract(self):
        for direction,node in zip(self.directions,self.nodes):
            is_leave=node.is_leave
            value=node.ast_value if is_leave else node.ast_type

            yield (is_leave,direction,value)


    @property
    def head(self):
        return self.nodes[0]

    @property
    def tail(self):
        return self.nodes[-1]

    @property
    def head_value(self):
        return self.head.ast_value

    @property
    def tail_value(self):
        return self.tail.ast_value

class CommonFeatureSet:
    def __init__(self,ast_object:Ast,path="") -> None:
        self.ast_object=ast_object

        self._leave_nodes=[node for node in self.ast_object.nodes if node.is_leave]
        self._not_leave_nodes=[node for node in self.ast_object.nodes if not node.is_leave]
        self._tokens=[node.ast_value for node in self._leave_nodes]
        self.path=path
    
    def search_nodes(self,node_types,node_value=None):
        return self.ast_object.search_nodes(node_types,node_value)
    
    def abstract(self):
        return f'{self.path} size: {len(self.ast_object.nodes)}'
    
    @property
    def leave_nodes(self):
        return self._leave_nodes
    
    @property
    def not_leave_nodes(self):
        return self._not_leave_nodes
    
    @property
    def nodes(self):
        return self.ast_object.nodes

    @property
    def tokens(self):
        return self._tokens
    
    @property
    def ast_edges(self):
        return self.ast_object.edges

    @staticmethod
    def from_file(path,lang=None):
        
        detect_lang=ProgramLanguage.match(path).value
        
        if lang is not None and detect_lang!=lang:
            return None
        
        source=feature_api.read_bytes(path)
        lang=ProgramLanguage.match(path).value
        
        if lang==ProgramLanguage.C.value:
            lang=ProgramLanguage.Cpp.value
        
        if lang==ProgramLanguage.Others.value:
            return None

        return CommonFeatureSet(feature_api.create_ast_obj(source,lang),path)

