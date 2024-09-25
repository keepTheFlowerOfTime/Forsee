from __future__ import annotations
from tree_sitter import Language,Parser,Node
from collections import deque

import codesecurity.feature.objects as feature_objects
import os

ParserInstance={

}

class TreesitterAdapter:
    @staticmethod
    def astv1(root_node):
        ast_edge = []
        # 获取父节点的下标，然后添加父节点到子节点的边的连接。
        # 使用一个父节点的stack

        traversal_stack = [root_node]
        traversal_index_stack = [None]
        count = -1

        nodes=[]

        while traversal_stack:
            count += 1
            cur_node = traversal_stack.pop()
            cur_node_parent_index = traversal_index_stack.pop()
            nodes.append(cur_node)
            if cur_node_parent_index != None:
                ast_edge.append([cur_node_parent_index, count,cur_node.type])

            if len(cur_node.children) == 0 and cur_node.type != 'comment':
                # 叶子节点，没有子节点了。
                continue
            else:
                traversal_index_stack.extend([count] * len(cur_node.children))
                traversal_stack.extend(cur_node.children[::-1])  # 倒序入栈，保证先序遍历。
        
        return nodes,ast_edge

    @staticmethod
    def astv2(root_node):
        ast_edge = []
        # 获取父节点的下标，然后添加父节点到子节点的边的连接。
        # 使用一个父节点的stack

        traversal_stack = deque([root_node])
        traversal_index_stack = deque([None])
        count = -1

        nodes=[]

        while traversal_stack:
            count += 1
            cur_node = traversal_stack.pop()
            cur_node_parent_index = traversal_index_stack.pop()
            nodes.append(cur_node)
            if cur_node_parent_index != None:
                ast_edge.append([cur_node_parent_index, count,cur_node.type])

            if len(cur_node.children) == 0 and cur_node.type != 'comment':
                # 叶子节点，没有子节点了。
                continue
            else:
                traversal_index_stack.extend([count] * len(cur_node.children))
                traversal_stack.extend(cur_node.children[::-1])  # 倒序入栈，保证先序遍历。
        
        return nodes,ast_edge

    @staticmethod
    def ast_fullv2(root_node):
        ast_edge = []
    # 获取父节点的下标，然后添加父节点到子节点的边的连接。
    # 使用一个父节点的stack

        traversal_stack = deque([root_node])
        traversal_parent_stack = deque([None])

        nodes=[]

        while traversal_stack:
            cur_node = traversal_stack.pop()
            cur_node_parent = traversal_parent_stack.pop()
            nodes.append(cur_node)
            if cur_node_parent is not None:
                ast_edge.append([cur_node_parent, cur_node])
            
            if len(cur_node.children) == 0  and cur_node.type != 'comment':
                # 叶子节点，没有子节点了。
                continue
            else:
                traversal_parent_stack.extend([cur_node] * len(cur_node.children))
                traversal_stack.extend(cur_node.children[::-1])  # 倒序入栈，保证先序遍历。
        
        return nodes,ast_edge

    @staticmethod
    def ast_fullv1(root_node):
        ast_edge = []
    # 获取父节点的下标，然后添加父节点到子节点的边的连接。
    # 使用一个父节点的stack

        traversal_stack = [root_node]
        traversal_parent_stack = [None]

        nodes=[]

        while traversal_stack:
            cur_node = traversal_stack.pop()
            cur_node_parent = traversal_parent_stack.pop()
            nodes.append(cur_node)
            if cur_node_parent is not None:
                ast_edge.append([cur_node_parent, cur_node])
            
            if len(cur_node.children) == 0  and cur_node.type != 'comment':
                # 叶子节点，没有子节点了。
                continue
            else:
                traversal_parent_stack.extend([cur_node] * len(cur_node.children))
                traversal_stack.extend(cur_node.children[::-1])  # 倒序入栈，保证先序遍历。
        
        return nodes,ast_edge

    @staticmethod
    def ast(root_node):
        return TreesitterAdapter.astv2(root_node)
    
    @staticmethod
    def ast_full(root_node):
        return TreesitterAdapter.ast_fullv2(root_node)

def build_library():
    exists_handle=lambda x: os.path.exists(x)
    lib_modules=[
        'libs/tree-sitter-c',
        'libs/tree-sitter-java',
        'libs/tree-sitter-cpp',
        'libs/tree-sitter-javascript'
    ]
    
    lib_modules=[x for x in lib_modules if exists_handle(x)]
    
    Language.build_library(
    'libs/language_parser.so',
    lib_modules
    )

def create_parser(lang='c')->Parser:
    
    if lang=='js': lang='javascript'
    
    build_library()
    
    if lang not in ParserInstance:
        language=Language("libs/language_parser.so",lang)
        
        result=Parser()
        result.set_language(language)

        ParserInstance[lang]=result

    
    return ParserInstance[lang]

def ast(root_node):
    return TreesitterAdapter.ast(root_node)

def ast_full(root_node):
    return TreesitterAdapter.ast_full(root_node)

def create_ast_object(source,lang):
    def create_ast_node(node:Node):
        text=""
        if isinstance(node.text,str):
            text=node.text
        elif isinstance(node.text,bytes):
            text=node.text.decode(encoding='utf-8',errors='ignore')
        
        content=dict(
            is_leave=node_is_leave(node),
            ast_type=node.type,
            ast_value=text,
            child_number=node.child_count
        )
        
        return feature_objects.AstNode(**content)
    
    def create_ast_edge(edge):
        content=dict(
            start_index=edge[0],
            end_index=edge[1]
        )
        
        return feature_objects.AstEdge(**content)
    
    parser=create_parser(lang)
    tree=parser.parse(source)
    root_node=tree.root_node
    nodes,ast_edge=ast(root_node)


    content=dict(
        nodes=list(map(create_ast_node,nodes)),
        edges=list(map(create_ast_edge,ast_edge))
    )
    
    return feature_objects.Ast(**content)
    
def tree_to_node_list(root_node):
    # code tokens 存储的是代码 中 每个token对应的起始位置和终止位置的集和。
    traversal_stack = [root_node]
    node_list = list()
    while traversal_stack:
        cur_node = traversal_stack.pop()
        node_list.append(cur_node)
        if (len(cur_node.children)==0 or cur_node.type=='string') and cur_node.type!='comment':
            continue
        else:
            traversal_stack.extend(cur_node.children[::-1]) # 倒序入栈，保证先序遍历。
        
    return node_list

def index_to_code_token(index,code):
    """
    Given a start and end index, return the corresponding code token
    
    Args:
      index: a tuple of two tuples, each of which is a tuple of two integers.
      code: the code to be tokenized
    
    Returns:
      A string of code.
    """
    # 根据code，将index映射为code。
    start_point=index[0]
    end_point=index[1]
    if start_point[0]==end_point[0]:
        s=code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s=""
        s+=code[start_point[0]][start_point[1]:]
        for i in range(start_point[0]+1,end_point[0]):
            s+=code[i]
        s+=code[end_point[0]][:end_point[1]]   
    return s

def tree_to_token_index(root_node):
    # 遍历整颗树，获取每个叶子节点（非注释节点）的（起始位置、终止位置）。
    # code tokens 存储的是代码 中 每个token对应的起始位置和终止位置的集和。
    if (len(root_node.children)==0 or root_node.type=='string') and root_node.type!='comment':
        return [(root_node.start_point,root_node.end_point)]
    else:
        code_tokens=[]
        for child in root_node.children:
            code_tokens+=tree_to_token_index(child)
        return code_tokens

def extract_ncs_flow(root_node):
    # 获取ncs边
    # 获取所有的AST Node，按照先序深度优先遍历，然后连接各个节点
    # 然后添加节点到子节点的边
    node_list = tree_to_node_list(root_node)
    ncs_edge = []
    # 先序深度优先遍历感觉像。
    # 所以NCS就是各个节点顺序连接。
    pre_node_range = range(len(node_list) - 1)
    next_node_range = range(1, len(node_list))
    ncs_edge = list(zip(pre_node_range, next_node_range))

    return ncs_edge

def get_node_depth(node,max_depth=64,cancel_warning=False):
    parent=node
    depth=0
    while parent is not None:
        if parent.parent==parent: break
        parent=parent.parent
        depth+=1

        if depth>max_depth and not cancel_warning:
            print(f'warn: unnormal depth for {parent}')

        if depth>max_depth:
            break
    return depth

def node_is_leave(node:Node):
    if node.children is None:
        return True
    elif node.child_count==0:
        return True
    
    return False

# def generate_property_for_tree_nodes(root:Node):
#     result={}
#     node_list=tree_to_node_list(root)
#     leave_nodes=[]
#     for i,node in enumerate(node_list):
#         node_depth=get_node_depth(node)
#         is_leaves=(len(node.children)==0 or node.type=='string') and node.type!='comment'
#         if is_leaves:
#             leave_nodes.append(node)
#         node_property=NodeProperty(node=node,depth=node_depth,is_leave=is_leaves,origin_index=i)
#         result[node.id]=node_property


#     #encoding=chardet.detect(root.text)['encoding']

#     for i,node in enumerate(leave_nodes):
#         result[node.id].mapping_index=i
    
    
#     virtual_index=0
#     for node in node_list:
#         node_property=result[node.id]
#         if node_property.is_leave:
#             continue
#         else:
#             node_property.mapping_index=virtual_index
#             virtual_index+=1
    
#     return result
    
# def generate_preprocess_context(trees:list[Tree],out_file="preprocessContext.json"):
#     edge_type_enums={}
#     edge_type_counter:Counter=Counter()
#     node_type_enums={}
#     node_type_counter:Counter=Counter()
#     for tree in trees:
#         nodes,edges=ast_full(tree.root_node)
#         for start_node,end_node in edges:
#             edge_type=EdgeType(start_node.type,end_node.type)
#             edge_type_counter.update([edge_type.id])
        
#         for node in nodes:
#             node_type_counter.update([node.type])
            
#     for key,_ in edge_type_counter.most_common():
#         edge_type_enums[key]=len(edge_type_enums)+2
    
#     for key,_ in node_type_counter.most_common():
#         node_type_enums[key]=len(node_type_enums)+2
    
#     context_obj=PreprocessContext(edge_type_counter={k:edge_type_counter[k] for k in edge_type_counter},
#                                   node_type_counter={k:node_type_counter[k] for k in node_type_counter},
#                                   edge_type_ids=edge_type_enums,
#                                   node_type_ids=node_type_enums)
    
#     context_obj_dict=asdict(context_obj)
    
#     with open(out_file,'w',encoding='utf-8') as f:
#         json.dump(context_obj_dict,f,ensure_ascii=False,indent=4)
    
#     return context_obj

# if __name__=="__main__":
#     c_language=Language("libs/language_parser.so",'c')

#     parser=Parser()
#     parser.set_language(c_language)

#     source_files=[os.path.join('test_data/CVE-TEST',e) for e in os.listdir('test_data/CVE-TEST')]

#     data=read_bytes(*source_files)
#     # 256实节点+256虚节点 不足就补齐
#     generate_preprocess_context([parser.parse(e) for e in data])
# for e in data:
#     tree=parser.parse(e)
#     root=tree.root_node
#     nodes,ast_edge=ast(root)
#     type_set=set([e.type for e in nodes])
#     p= lp.tokenize_code(root,e.decode('utf-8'))
#     p2=tree_to_node_list(root)
#     print(len(p),len(nodes))
#     print(len(tree_to_token_index(root)))
    
#     for e in p2:
#         #print(e.type,e.text)
#         print(get_node_depth(e))
    
    # for node in tree_to_node_list(root):
    #     print(node.text)
    #print(tree_to_token_index(root))

    # for e in type_set:
    #     print(e)

    # for node in nodes:
    #     print(node.text)