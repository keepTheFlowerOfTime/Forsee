from codesecurity.feature.objects import Ast, AstNode,AstPath,AstDirection

import random

class AstImageBuilder:
    def __init__(self,ast_object:Ast) -> None:
        self.ast_object=ast_object
        self.nearby_dict,self.nearby_dict_reverse=ast_object.get_nearby_matrix()
        self.node2index={node:i for i,node in enumerate(ast_object.nodes)}

        self.search_caches={}
    
    def search_nodes(self,node_types,use_caches=False):
        if isinstance(node_types,str):
            node_types=[node_types]
        
        if not use_caches:
            return self.ast_object.search_nodes(node_types)
        else:
            ret=[]
            for node_type in node_types:
                if node_type not in self.search_caches:
                    match_nodes=self.ast_object.search_nodes(node_type)
                    self.search_caches[node_type]=match_nodes
                    ret.extend(match_nodes)
                else:
                    ret.extend(self.search_caches.get(node_type,[]))
        
            return ret
    
    def dependency_in(self,node,anchor_types=[],force_anchor=['program']):
        if isinstance(node,AstNode):
            node_index=self.node2index[node]
        
        while True:
            in_nodes=self.nearby_dict_reverse.get(node_index,[])
            if len(in_nodes)==0:
                break
            
            node_index=in_nodes[0]
            node=self.ast_object.nodes[node_index]
            
            if node.ast_type in anchor_types:
                return node
            
            if node.ast_type in force_anchor:
                return node
        
        return None 
    
    def node_in(self,node):
        if isinstance(node,AstNode):
            node=self.node2index[node]
        
        in_nodes=self.nearby_dict_reverse.get(node,[])
        
        return [self.ast_object.nodes[i] for i in in_nodes]
    
    def node_out(self,node):
        if isinstance(node,AstNode):
            node=self.node2index[node]
        
        out_nodes=self.nearby_dict.get(node,[])    
            
        return [self.ast_object.nodes[i] for i in out_nodes]
    
    def node_sibling(self,node):
        if isinstance(node,AstNode):
            node=self.node2index[node]
        
        in_nodes=self.nearby_dict_reverse.get(node,[])
        
        
        assert len(in_nodes)==1
        
        out_nodes=self.nearby_dict.get(in_nodes[0],[])
        
        return [self.ast_object.nodes[i] for i in out_nodes]
    
    def get_leave_node_indexes(self):
        for i in range(len(self.ast_object.nodes)):
            node=self.ast_object.nodes[i]
            if node.is_leave:
                yield i
    
    def build_all_node_depth(self):
        node2depth={}
        
        target_indexes=[0]
        previous_depths=[0]
        while len(target_indexes)>0:
            target_index=target_indexes.pop()
            target_depth=previous_depths.pop()
            node2depth[target_index]=target_depth
            
            connect_indexs=self.nearby_dict.get(target_index,[])
            target_indexes.extend(connect_indexs)
            previous_depths.extend([target_depth+1]*len(connect_indexs))

        return node2depth
    
    def get_node_neighbor_out(self,node_index,distance=1,include_self=True):
        if distance==0:
            if include_self:
                return [node_index]
            else:
                return []
        
        neighbors=self.nearby_dict.get(node_index,[])
        if distance==1:  
            if include_self:
                return [node_index]+neighbors
            else:
                return neighbors
        
        result=[]
        if include_self:
            result.append(node_index)
            
        for neighbor in neighbors:
            result.extend(self.get_node_neighbor_out(neighbor,distance-1))
        
        return result
    
    def get_node_neighbor_in(self,node_index,distance=1,include_self=True):
        if distance==0:
            if include_self:
                return [node_index]
            else:
                return []
        
        neighbors=self.nearby_dict_reverse.get(node_index,[])
        if distance==1:
            if include_self:
                return [node_index]+neighbors
            else:
                return neighbors
        
        result=[]
        if include_self:
            result.append(node_index)
            
        for neighbor in neighbors:
            result.extend(self.get_node_neighbor_in(neighbor,distance-1))
        
        
        return result
    
    def walk(self,start_index,end_index):
        start2root=self.walk2root(start_index)
        end2root=self.walk2root(end_index)

        path=[]
        directions=[AstDirection.start]
        
        dev=0
        for i in range(min(len(start2root),len(end2root))):
            i=-i-1
            if start2root[i]==end2root[i]:        
                dev=i
        
        turn2end=end2root[:dev]
        turn2end.reverse()        
        
        path.extend(start2root[:dev])
        directions.extend([AstDirection.Up]*(dev-1))
        path.extend(turn2end)
        directions.extend([AstDirection.Down]*(len(turn2end)-1))        
        
        nodes=[self.ast_object.nodes[i] for i in path]

        return AstPath(nodes,directions)
    # def walk2leave(self,node_index):
    #     paths=[node_index]
    #     while node_index in self.nearby_dict:
    #         node_index=random.choice(self.nearby_dict[node_index])
    #         paths.append(node_index)
        
    #     return paths
        
    def walk2root(self,node_index):
        if isinstance(node_index,AstNode):
            node_index=self.node2index[node_index]
            
        path=[node_index]
        while node_index in self.nearby_dict_reverse:
            node_index=self.nearby_dict_reverse[node_index][0]
            path.append(node_index)
        
        return path
    
    @property
    def leaves_number(self):
        return len(self.ast_object.nodes)-len(self.nearby_dict)

