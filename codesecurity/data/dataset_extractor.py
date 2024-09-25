
import os
import shutil
import hashlib
import torch
import random
import re



def extract_gcj_cpp(data_dir,target_dir):
    def check_code_dir(name):
        if 'code' in name:
            return True
        else:
            return False
    def check_json_dir(name):
        if 'json' in name:
            return True
        else:
            return False
    
    allow_ext=['.cpp']
    
    def info(file_name:str):
        task_id,author_id=file_name.split('_')
    
        return author_id,task_id
    code_dirs=[]
    json_dirs=[]
    
    all_dirs=os.listdir(data_dir)
    
    for d in all_dirs:
        full_d=os.path.join(data_dir,d)
        if os.path.isdir(full_d):
            if check_code_dir(full_d):
                code_dirs.append(full_d)
            if check_json_dir(full_d):
                json_dirs.append(full_d)
    
    
    author_mapping={}            
    for code_dir in code_dirs:
        files=os.listdir(code_dir)            
        for file in files:
            file_name,ext=os.path.splitext(file)
            if ext in allow_ext:
                author_id,task_id=info(file_name)
            
                if author_id not in author_mapping:
                    author_mapping[author_id]=[]
                
                author_mapping[author_id].append(os.path.join(code_dir,file))
    
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    
    for author_id in author_mapping:
        author_dir=os.path.join(target_dir,author_id)
        if not os.path.exists(author_dir):
               os.mkdir(author_dir)
        
        for src_file in author_mapping[author_id]:
            shutil.copy(src_file,author_dir)

def collect_author_file_pair(data_dir,author_dir_handle=None):

    if author_dir_handle is None:
        author_dir_handle=lambda x:x

    all_dirs=os.listdir(data_dir)
    
    author_mapping={}        

    all_dirs=os.listdir(data_dir)
    
    author_mapping={}        

    for d in all_dirs:

        author_dir=author_dir_handle(os.path.join(data_dir,d))

        files=os.listdir(author_dir)

        for f in files:
            if d not in author_mapping:
                author_mapping[d]=[]
            author_mapping[d].append(os.path.join(data_dir,d,f))

    return author_mapping

def extract_java40(data_dir,target_dir):
    all_dirs=os.listdir(data_dir)
    
    author_mapping={}        

    for d in all_dirs:
        full_d=os.path.join(data_dir,d)
        content=os.listdir(full_d)
        if len(content)==1:
            files=os.listdir(os.path.join(full_d,content[0]))
            for f in files:
                if d not in author_mapping:
                    author_mapping[d]=[]
                author_mapping[d].append(os.path.join(full_d,content[0],f))

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    for author_id in author_mapping:
        author_dir=os.path.join(target_dir,author_id)
        if not os.path.exists(author_dir):
            os.mkdir(author_dir)
        
        for src_file in author_mapping[author_id]:
            shutil.copy(src_file,author_dir)

def random_select_java40(data_dir,target_dir,author_number=20,number=2,clear=False):
    if clear:
        shutil.rmtree(target_dir)
    all_dirs=os.listdir(data_dir)
    
    author_mapping={}        

    for d in all_dirs:
        files=os.listdir(os.path.join(data_dir,d))
        for f in files:
            if d not in author_mapping:
                author_mapping[d]=[]
            author_mapping[d].append(os.path.join(data_dir,d,f))

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    authors=list(author_mapping.keys())
    random.shuffle(authors)

    select_author=[e for e in authors if '.' not in e]
    select_author=select_author[:min(len(select_author),author_number)]

    for author_id in select_author:
        author_dir=os.path.join(target_dir,author_id)
        if not os.path.exists(author_dir):
            os.mkdir(author_dir)
        


        select_data=random.sample(author_mapping[author_id],number)

        for src_file in select_data:
            shutil.copy(src_file,author_dir)


def random_split(dataset_dir,target_dir,group={'train':0.7,'validate':0.15,'test':0.15}):
    keys=list(group.keys())
    all_dirs=os.listdir(dataset_dir)
    all_dirs=[e for e in all_dirs if os.path.isdir(os.path.join(dataset_dir,e))]
    author_mapping={}        

    for d in all_dirs:
        files=os.listdir(os.path.join(dataset_dir,d))
        for f in files:
            if d not in author_mapping:
                author_mapping[d]=[]
            author_mapping[d].append(os.path.join(dataset_dir,d,f))

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    for key in keys:
        key_dir=os.path.join(target_dir,key)
        if not os.path.exists(key_dir):
            os.mkdir(key_dir)

    for author_id in author_mapping:
        group_samples=author_mapping[author_id]
        random.shuffle(group_samples)
        start=0
        for i,key in enumerate(keys):
            author_group_dir=os.path.join(target_dir,key,author_id)
            if not os.path.exists(author_group_dir):
                os.mkdir(author_group_dir)
            
            if i==len(keys)-1:
                select_data=group_samples[start:]
            else:
                select_data=group_samples[start:start+int(len(group_samples)*group[key])]
            start+=int(len(group_samples)*group[key])

            for src_file in select_data:
                shutil.copy(src_file,author_group_dir)

def target_attack_info(path:str):
    pattern=""
    ext=os.path.splitext(path)[-1]
    pattern=f'(.*?)##(.*?)###(.+?){ext}'
    
    value= re.search(pattern,path)

    if value is None:
        return None
    
    if len(value.groups())==3:
        name,origin_author,target_author=value.groups()
    
        return f'{name}{ext}',origin_author,target_author
    
    else:
        return None
    
def count_file_line(path):
    with open(path,'r',errors='ignore') as f:
        result=0
        for l in f:
            result+=1 
        return result
    

def ropgen_robust_data_select(robust_data_dir,origin_data_dir,target_dir,threshold=20,mode='target'):
    robust_mapping=collect_author_file_pair(robust_data_dir)

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    for author in robust_mapping:
        for file in robust_mapping[author]:
            name=os.path.basename(file)

            attack_info=target_attack_info(name)
            if attack_info is None:
                continue

            name,origin_author,target_author=attack_info

            origin_line=count_file_line(os.path.join(origin_data_dir,origin_author,name))
            robust_line=count_file_line(file)

            if abs(origin_line-robust_line)>=threshold:
                continue

            
            if robust_line<10:
                continue

            new_author_dir=os.path.join(target_dir,target_author)

            if mode=='untarget':
                new_author_dir=os.path.join(target_dir,origin_author)

            if not os.path.exists(new_author_dir):
                os.mkdir(new_author_dir)
            shutil.copy(file,new_author_dir)

def repeat_line_in_dataset_per_author(data_dir):
    def get_repeat_line(contents,start=0):
        common_line_number=0
        
        if len(contents)==0: return 0

        compare=contents[0]
        for i,line in enumerate(compare):
            if i<start: continue
            for e in contents:
                if i>=len(e) or line != e[i]:
                    return common_line_number

            common_line_number+=1

        return common_line_number

    data = collect_author_file_pair(data_dir)

    abstracts=[]

    for author in data:
        contents=[]
        for path in data[author]:
            with open(path,'rb') as f:
                contents.append([l for l in f])
        
        abstracts.append((author,max(get_repeat_line(contents),get_repeat_line(contents,1),get_repeat_line(contents,2))))

    return abstracts

def dataset_abstract(dataset_dir,dataset_name):
    import csv
    csv_file=open(f'{dataset_name}.csv','w')
    csv_writer=csv.writer(csv_file)

    csv_writer.writerow(['path','length','line_number','dataset_name','label'])

    for root,dirs,files in os.walk(dataset_dir):
        for file in files:  
            with open(os.path.join(root,file),'rb') as f:
                contents=f.read().splitlines()
            
                line_number=len(contents)
                length=sum([len(e) for e in contents])
                label=os.path.join(root,file).split('/')[-2]

                dataset_name=dataset_name
                csv_writer.writerow([os.path.join(root,file),length,line_number,dataset_name,label])

    csv_file.close()

if __name__=="__main__":
    #random_split('data/java40','data/java40_split')
    random_select_java40('data/a_github_c','/home/passwd123/XiaoweiGuo/vscode/code-security/generate_data/a_github_c/program_file/test',20,5,True)
    random_select_java40('data/a_github_c','/home/passwd123/XiaoweiGuo/vscode/code-security/generate_data/a_github_c/program_file/target_author_file',20,5,True)
    #ropgen_robust_data_select('RoPGen/src/coding style attacks/program_file/targeted_attack_file','RoPGen/src/coding style attacks/program_file/target_author_file','data/gcj_cpp_mislead',20,'target')
    #ropgen_robust_data_select('RoPGen/src/coding style attacks/program_file/untargeted_attack_file','RoPGen/src/coding style attacks/program_file/target_author_file','data/gcj_cpp_untarget_best',20,'untarget')
    #print(target_attack_info('BaseDaoImpl##chweixin###applewjg.java'))
    #extract_java40('data/pack/40authors','data/java40')
    # abstracts=repeat_line_in_dataset_per_author('data/gcj_cpp')

    # count_handle=lambda number:[e[0] for e in abstracts if e[1]>=number]


    # count3=len(count_handle(3))
    # count5=len(count_handle(5))
    # count10=len(count_handle(10))
    
    # print(f'{count3}/{len(abstracts)}')
    # print(f'{count5}/{len(abstracts)}')
    # print(f'{count10}/{len(abstracts)}')

    #dataset_abstract('/home/passwd123/XiaoweiGuo/vscode/code-semantic-classcification/data/gcj_cpp','gcj_cpp')