import codesecurity.feature.tree_sitter_adapter as ts_back



def create_ast_obj(source,lang):
    #print(lang)
    return ts_back.create_ast_object(source,lang)

def read_bytes(file_name:str):
    with open(file_name,'rb') as f:
        return f.read(-1)