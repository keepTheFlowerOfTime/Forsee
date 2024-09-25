import tarfile
import shutil
import os

def iter_dir(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
                file_path = os.path.join(root, file)
                # 处理筛选出来的文件，例如输出文件路径
                yield file_path

def extract_tgz(path,target_dir=None):
    # 打开tgz文件
    if target_dir is None:
        target_dir=os.path.dirname(path)
    
    
    if os.path.exists(target_dir) is False:
        os.makedirs(target_dir)
        
    with tarfile.open(path, 'r:gz', encoding='utf-8') as tar:
        
        # 解压所有文件到指定目录下
        tar.extractall(path=target_dir)

def peek_tgz(path):
    with tarfile.open(path, 'r:gz', encoding='utf-8') as tar:
        tar_infos=tar.getmembers()
        for tarinfo in tar_infos:
            print(tarinfo.name)

        return tar_infos
    
def list_npm(package_dir,filter=['.js']):
    def filter_handle(path:str):
        for c in filter:
            if not path.endswith(c):
                return False
        return True
    
    all_paths=list(iter_dir(package_dir))
    fitter_path=[e for e in all_paths if filter_handle(e)]
    
    return fitter_path

def extract_npm_code_to_dir(package_dir,code_dir,filter=['.js']):
    
    exclude_names=['@']
    exclude_dirs=['node_modules']
    
    for e in exclude_names:
        if e in package_dir:
            return
    
    
    fitter_paths=list_npm(package_dir,filter=filter)
    
    if len(fitter_paths)==0:
        return
    
    if os.path.exists(code_dir) is False:
        os.makedirs(code_dir)
    
    for path in fitter_paths:
        target_path=path.replace(package_dir,code_dir)
        target_dir=os.path.dirname(target_path)
        
        for e in exclude_dirs:
            if e in target_dir:
                return
        
        if os.path.exists(target_dir) is False:
            os.makedirs(target_dir)
            
        shutil.copy(path,target_path)
    
