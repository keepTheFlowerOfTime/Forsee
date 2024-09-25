from codesecurity.feature.objects import Ast,AstNode,CommonFeatureSet
from codesecurity.feature.ast_image import AstImageBuilder
from codesecurity.feature.node_type_const import JSNodeTypes

import random
import sys

sys.setrecursionlimit(5000)

class JsValueDependency:
    # in means what var need the call function
    # out means call function need what var 
    
    @staticmethod
    def call_function_in(ast_node:AstNode,image_builder:AstImageBuilder):
        anchor_types=['variable_declarator','assignment_expression','augmented_assignment_expression','subscript_expression']
        return JsValueDependency.default_in(ast_node,image_builder,anchor_types)
        
    @staticmethod 
    def const_in(ast_node:AstNode,image_builder:AstImageBuilder):
        return JsValueDependency.default_in(ast_node,image_builder)
    
    @staticmethod
    def default_in(ast_node:AstNode,image_builder:AstImageBuilder,anchor_types=None):
        if anchor_types is None:
            anchor_types=['variable_declarator','assignment_expression','augmented_assignment_expression','call_expression']
        force_anchor_types=JSNodeTypes().statement_types()+["switch_body","subscript_expression"]
        anchor_node=image_builder.dependency_in(ast_node,anchor_types,force_anchor_types)
        
        if anchor_node is None or anchor_node.ast_type in force_anchor_types:
            return None,anchor_node if anchor_node else None
        
        if anchor_node.ast_type in ['variable_declarator']:
            return image_builder.node_out(anchor_node)[0],anchor_node
        elif anchor_node.ast_type in ['call_expression','assignment_expression','augmented_assignment_expression','subscript_expression']:
            return JsNodeSearch.search_first_identifier_in_children_js(anchor_node,image_builder),anchor_node
        else:
            return anchor_node,anchor_node
    
    @staticmethod
    def string_dependency_graph(ast_object:Ast,image_builder:AstImageBuilder):
        candidates=image_builder.search_nodes(['string'])
        
        nodes=[]
        edges=[]
        
        for node in candidates:
            in_node,anchor_type=JsValueDependency.default_in(node,image_builder)
            if in_node is None: continue
            
            nodes.append(node)
            edges.append([node,in_node])

        return nodes,edges
    
    @staticmethod
    def number_dependency_graph(ast_object:Ast,image_builder:AstImageBuilder):
        candidates=image_builder.search_nodes(['number'])
        
        nodes=[]
        edges=[]
        
        for node in candidates:
            in_node,anchor_type=JsValueDependency.default_in(node,image_builder)
            if in_node is None: continue
            
            nodes.append(node)
            edges.append([node,in_node])
            
        return nodes,edges
    
    @staticmethod
    def func_dependency_graph(ast_object:Ast,image_builder:AstImageBuilder):
        candidates,arguments=JsNodeSearch.search_all_call_function_js(ast_object,image_builder)
        
        nodes=[]
        edges=[]
        
        for func_handle,args in zip(candidates,arguments):
            in_node,anchor_type=JsValueDependency.call_function_in(func_handle,image_builder)
            
            func_identifier=JsNodeSearch.search_first_identifier_in_children_js(func_handle,image_builder)
            arg_identifiers=JsNodeSearch.search_core_identifier_in_children_js(args,image_builder)
            
            if in_node is None: continue
            
            nodes.append(func_identifier)
            edges.append([func_identifier,in_node])
            
            for arg_identifier in arg_identifiers:
                nodes.append(arg_identifier)
                edges.append([arg_identifier,func_identifier])
        
        func_nodes,func_fields,ret=JsNodeSearch.search_func_declaration_js(ast_object,image_builder)
        for func_node,func_field,ret_node in zip(func_nodes,func_fields,ret):
            identifier,parameter,statement_block=func_field
            
            param_identifiers=JsNodeSearch.search_core_identifier_in_children_js(parameter,image_builder)
            
            nodes.append(identifier)
            for param_identifier in param_identifiers:
                nodes.append(param_identifier)
                for ret_item in ret_node:
                    edges.append([param_identifier,ret_item])
            
            for ret_item in ret_node:
                nodes.append(ret_item)
                edges.append([ret_item,identifier])
            
            
        return nodes,edges

    @staticmethod
    def variable_mirror_graph(ast_object:Ast,image_builder:AstImageBuilder):
        
        nodes=image_builder.search_nodes(['identifier'])
        edges=[]
        mirrors={}
        for node in nodes:
            if node.ast_value not in mirrors:
                mirrors[node.ast_value]=[]
            mirrors[node.ast_value].append(node)
        
        for key,value in mirrors.items():
            if len(value)>1:
                for i in range(len(value)-1):
                    edges.append([value[i],value[i+1]])
            
        return nodes,edges
    
class JsNodeSearch:
    
    @staticmethod
    def search_return_in_block(ast_node:AstNode,image_builder:AstImageBuilder):
        choose_type=['return_statement']

        return_statements=JsNodeSearch.search_node_in_children_js(ast_node,image_builder,choose_type)
        
        return return_statements
    
    @staticmethod
    def search_func_declaration_js(ast_object:Ast,image_builder:AstImageBuilder):
        choose_type=["function"]
        nodes=image_builder.search_nodes(choose_type)
        
        nodes_in=[image_builder.node_in(e)[0] for e in nodes]
        
        normal_func_nodes=[]
        anoym_func_nodes=[]
        for node_in in nodes_in:
            if node_in.ast_type=='function_declaration':
                normal_func_nodes.append(node_in)
            elif node_in.ast_type=='function':
                anoym_func_nodes.append(node_in)
        
        normal_group_children=[image_builder.node_out(e) for e in normal_func_nodes]
        anoym_group_children=[image_builder.node_out(e) for e in anoym_func_nodes]
        
        statement_blocks=[e[-1] for e in normal_group_children]+[e[-1] for e in anoym_group_children]
        parameters=[e[-2] for e in normal_group_children]+[e[-2] for e in anoym_group_children]
        identifiers=[e[1] for e in normal_group_children]+anoym_func_nodes
        
        rets=[JsNodeSearch.search_return_in_block(e,image_builder) for e in statement_blocks]
        for i,ret in enumerate(rets):
            temp=[]
            for e in ret:
                temp+=JsNodeSearch.search_core_identifier_in_children_js(e,image_builder)
            rets[i]=temp
                
        return normal_func_nodes+anoym_func_nodes,list(zip(identifiers,parameters,statement_blocks)),rets
    
    @staticmethod
    def search_core_identifier_in_children_js(ast_node:AstNode,image_builder:AstImageBuilder):
        zero_gram_types=['identifier','literal','string','number','boolean','null','regexp','function']
        single_gram_types=['subscript_expression','call_expression']
        multi_gram_types=['binary_expression','unary_expression','arguments']
        children=image_builder.node_out(ast_node)

        if ast_node.ast_type in zero_gram_types:
            return [ast_node]
        
        elif ast_node.ast_type in multi_gram_types:
            ret=[]
            for child in children:
                ret+=JsNodeSearch.search_core_identifier_in_children_js(child,image_builder)
            return ret
        
        else:
            identifier=JsNodeSearch.search_first_identifier_in_children_js(ast_node,image_builder)
            if identifier is None:
                return []
            else:
                return [identifier]
    
    @staticmethod
    def search_node_in_children_js(ast_node:AstNode,image_builder:AstImageBuilder,select_node_types:list):

        children=image_builder.node_out(ast_node)
        
        if len(children)==0:
            return []

        ret=[]
        for child in children:
            if child.ast_type in select_node_types:
                ret.append(child)
            else:
                ret+=JsNodeSearch.search_node_in_children_js(child,image_builder,select_node_types)
        return ret
    
    @staticmethod
    def search_first_node_in_children_js(ast_node:AstNode,image_builder:AstImageBuilder,select_node_types:list):
        children=image_builder.node_out(ast_node)
        if len(children)==0: return None
        for child in children:
            if child.ast_type in select_node_types:
                return child
            else:
                ret=JsNodeSearch.search_first_node_in_children_js(child,image_builder,select_node_types)
                if ret is not None:
                    return ret
        return None
    
    @staticmethod
    def search_first_identifier_in_children_js(ast_node:AstNode,image_builder:AstImageBuilder):
        select_types=['identifier']
        return JsNodeSearch.search_first_node_in_children_js(ast_node,image_builder,select_types)
    
    # @staticmethod
    # def search_all_string_js(ast_object:Ast):
    #     js_types=JSNodeTypes()

    #     choose_type=["string"]

    #     nodes=ast_object.search_nodes(choose_type)
        
    #     return nodes

    # @staticmethod
    # def search_all_number_js(ast_object:Ast):
    #     js_types=JSNodeTypes()

    #     choose_type=["number"]

    #     nodes=ast_object.search_nodes(choose_type)
        
    #     return nodes

    @staticmethod
    def search_all_call_function_js(ast_object:Ast,image_builder:AstImageBuilder):
        js_types=JSNodeTypes()

        choose_type=["call_expression"]

        nodes=image_builder.search_nodes(choose_type)
        group_children=[image_builder.node_out(e) for e in nodes]
        
        func_identifiers=[e[0] for e in group_children]
        arugments=[e[1] for e in group_children]
        
        return func_identifiers,arugments
    
    @staticmethod
    def search_assignment_js(ast_object:Ast,image_builder:AstImageBuilder):
        choose_type=['assignment_expression','augmented_assignment_expression']
        nodes=image_builder.search_nodes(choose_type)
        nodes_children=[image_builder.node_out(node) for node in nodes]
        
        # for i in range(len(nodes_children)):
        #     left=nodes_children[i][0]
        #     right=nodes_children[i][-1]
            
        #     left_identifier=left if left.ast_type=='identifier' else search_first_identifier_in_children_js(left,image_builder)
        #     right_identifier=right if right.ast_type=='identifier' else search_first_identifier_in_children_js(right,image_builder)
            
        #     nodes_children[i]=[left_identifier,right_identifier]

        return nodes,nodes_children
    
    @staticmethod
    def identifier2identifier_graph(ast_obj:Ast,image_builder:AstImageBuilder):
        assignment_nodes,assignment_children=JsNodeSearch.search_assignment_js(ast_obj,image_builder)
        
        nodes=[]
        edges=[]
        
        for node,child in zip(assignment_nodes,assignment_children):
            left,right=child[0],child[-1]
            if left is None or right is None: continue
            
            left_identifier=JsNodeSearch.search_first_identifier_in_children_js(left,image_builder)
            right_idenfifiers=JsNodeSearch.search_core_identifier_in_children_js(right,image_builder)
            
            right_idenfifiers=[e for e in right_idenfifiers if e is not None]
            
            if left_identifier is None:
                continue
            
            nodes.append(left_identifier)
            nodes+=right_idenfifiers
            
            for right_idenfifier in right_idenfifiers:
                edges.append([left_identifier,right_idenfifier])
                
        return nodes,edges

def build_const2identifier_graph_js(image_builder:AstImageBuilder):
    _,string_dependency=JsValueDependency.string_dependency_graph(image_builder.ast_object,image_builder)
    _,number_dependency=JsValueDependency.number_dependency_graph(image_builder.ast_object,image_builder)
    _,func_dependency=JsValueDependency.func_dependency_graph(image_builder.ast_object,image_builder)
    

    return image_builder.ast_object.nodes,string_dependency+number_dependency+func_dependency

def build_identifier2identifier_graph_js(image_builder:AstImageBuilder):
    _,identifier_dependency=JsNodeSearch.identifier2identifier_graph(image_builder.ast_object,image_builder)


    return image_builder.ast_object.nodes,identifier_dependency

def build_identifier_mirror_graph_js(image_builder:AstImageBuilder):
    _,mirror_dependency=JsValueDependency.variable_mirror_graph(image_builder.ast_object,image_builder)

    return image_builder.ast_object.nodes,mirror_dependency

def list_all_node_types(ast:Ast):
    node_types=set()
    for node in ast.nodes:
        node_types.add(node.ast_type)
    return node_types

def list_all_node_values(ast:Ast):
    node_values=set()
    for node in ast.nodes:
        node_values.add(node.ast_value)
        
    return node_values

def list_child_node(image_builder:AstImageBuilder,node_index):
    neighbor_indexs=image_builder.get_node_neighbor_out(node_index,2)

    neighbor_nodes=[image_builder.ast_object.nodes[index] for index in neighbor_indexs]
    
    return [(node.ast_type,node.ast_value) for node in neighbor_nodes]
    
def list_walk_path(path,features:CommonFeatureSet):
    path_texts=[]
    for node_index in path:
        node=features.nodes[node_index]
        path_texts.append(node.text)
        
    return path_texts

def random_walk(features:CommonFeatureSet):
    start_node,end_node=random.sample(features.leave_nodes,2)

    image_builder=AstImageBuilder(features.ast_object)
    path=image_builder.walk(features.ast_object.nodes.index(start_node),features.ast_object.nodes.index(end_node))
    
    return path

def get_anchor_node_js(features:CommonFeatureSet):
    anchor_type=['number','string_fragment']
    
    nodes=[]
    
    for node in features.leave_nodes:
        if node.ast_type in anchor_type:
            nodes.append(node)
            
    return nodes

def get_anchor_node_vision_js():
    vision={
        'number':3,
        'string_fragment':4,
    }
    
    return vision

def get_anchor_node_neighbor(node:AstNode,image_builder:AstImageBuilder):
    node_visions=get_anchor_node_vision_js()
    
    vision=node_visions.get(node.ast_type,0)
    
    neighbor_nodes=image_builder.get_node_neighbor_in(image_builder.node2index[node],vision)
    
    return neighbor_nodes

def get_anchor_node_variable(node:AstNode,image_builder:AstImageBuilder):
    
    search_parent_type=['variable_declaration']
    search_type=['identifier']
    
    node_visions=get_anchor_node_vision_js()
    
    vision=node_visions.get(node.ast_type,0)
    
    neighbor_nodes=image_builder.get_node_neighbor_in(image_builder.node2index[node],vision)
    
    #neighbor_nodes_view=[image_builder.ast_object.nodes[neighbor] for neighbor in neighbor_nodes]
    
    neighbor_nodes=[neighbor for neighbor in neighbor_nodes if image_builder.ast_object.nodes[neighbor].ast_type in search_parent_type]
    
    child_nodes=[]
    for neighbor in neighbor_nodes:
        child_nodes+=[index for index in image_builder.get_node_neighbor_out(neighbor,vision) if image_builder.ast_object.nodes[index].ast_type in search_type]
    
    return set(child_nodes)
    
def list_anchor_variable(features:CommonFeatureSet,image_builder:AstImageBuilder):
    anchor_nodes=get_anchor_node_js(features)
    
    anchor_variables=[]
    
    for node in anchor_nodes:
        neighbors=get_anchor_node_neighbor(node,image_builder)
        
        for neighbor in neighbors:
            anchor_variables.append(neighbor.ast_value)
            
    return anchor_variables

def search_identifier_js(ast_object:Ast):
    search_type=['identifier']
    
    search_result={}
    
    for index in range(len(ast_object.nodes)):
        node=ast_object.nodes[index]
        if node.ast_type in search_type:
            if node.ast_value not in search_result:
                search_result[node.ast_value]=[]
            search_result[node.ast_value].append(index)
            
    return search_result




def search_fusion_candidate(image_builder:AstImageBuilder,threshold=16):
    identifier_list=search_identifier_js(image_builder.ast_object)

    total_number=sum([len(value) for value in identifier_list.values()])
    
    ratio=0.07
    
    threshold=min(threshold,int(total_number*ratio))
    
    fusion_candidate={}
    for key,value in identifier_list.items():
        if len(value)>=threshold:
            fusion_candidate[key]=value    
    
    return fusion_candidate

def list_fusion_node_state(features:CommonFeatureSet,image_builder:AstImageBuilder,threshold=12):
    anchor_nodes=get_anchor_node_js(features)

    anchor_variable_indexs=set()
    for node in anchor_nodes:
        neighbors=get_anchor_node_variable(node,image_builder)
        anchor_variable_indexs.update(neighbors)
    
    candidates=search_fusion_candidate(image_builder,threshold)
    #print(candidates)
    fusion_state={}
    for variable_index in anchor_variable_indexs:
        variable_node=image_builder.ast_object.nodes[variable_index]
        variable_name=variable_node.ast_value
        if variable_name not in candidates: continue
        if variable_name not in fusion_state:
            fusion_state[variable_name]=candidates[variable_name]
    
    return fusion_state

def serialize_node(image_builder:AstImageBuilder,node_index):
    def traversal(node_index):
        child_nodes=[traversal(child_index) for child_index in image_builder.get_node_neighbor_out(node_index,1,False)]
        return child_nodes
    
    ast_object=image_builder.ast_object
    
    return traversal(node_index)
    