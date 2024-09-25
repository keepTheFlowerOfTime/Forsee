import numpy as np
from scipy.sparse import lil_matrix

def random_walk_graph(nodes:list,edges:list,path_number:int,step_number:int,channel_number:int=1):
    """
    path_number: the number round of path generation
    step_number: the length of each path
    channel_number: for each start_node, we generate channel_number paths
    """
    
    #print(len(nodes))
    
    valid_start=set()
    matrix=lil_matrix((len(nodes),len(nodes)))
    for edge in edges:
        start,end=edge
        matrix[start,end]=1
        valid_start.add(start)
    
    
    
    ret=np.zeros([path_number,step_number,channel_number]) #32*32*3
    
    if len(valid_start)==0:
        return ret
    
    start_nodes=[np.random.choice(list(valid_start)) for i in range(path_number)]
    start_nodes.sort()
    
    for i in range(path_number):
        for j in range(channel_number):
            start_node=start_nodes[i]
            for k in range(step_number):
                #print(start_node.shape)
                ret[i][k][j]=start_node
                
                temp=matrix[start_node,:].toarray().flatten()
                #print(temp.shape)
                candidate_items=np.argwhere(temp>0)
                candidate_items=candidate_items[:,0]
                
                #print(candidate_items)
                
                if candidate_items.size==0:
                    break
                
                else:
                    start_node=np.random.choice(candidate_items)
                
    return ret