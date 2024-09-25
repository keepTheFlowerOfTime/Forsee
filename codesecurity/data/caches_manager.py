import pickle
import os
import codesecurity.data.api as data_api

class GroupPipe:
    def __init__(self,caches_path,group_max_number=25,addon=None) -> None:
        self.group_max_number=group_max_number
        self.group_buffer=[]
        self.group_number=0
        self.caches_path=caches_path
        self.addon=addon
    
    def add_group(self,obj):
        self.group_buffer.append(obj)

        if len(self.group_buffer)>=self.group_max_number:
            self.save_group()
    
    def add_group_list(self,obj_list):
        for obj in obj_list:
            self.add_group(obj)

    def save_group(self):
        group_path=self.group_name(self.group_number)
        with open(group_path,'wb') as f:
            pickle.dump(self.group_buffer,f)

        self.group_buffer=[]
        self.group_number+=1
        return group_path
    
    def save(self,addon=None):
        if addon is not None:
            self.addon=addon
        if len(self.group_buffer)>0:
            self.save_group()
        with open(self.caches_path,'wb') as f:
            pickle.dump(self,f)


    def group_name(self,number):
        name,ext=os.path.splitext(self.caches_path)

        return f'{name}{number}{ext}'
    
    def iter_group(self):
        start_group=0
        while True:
            if start_group>self.group_number: break
            group_path=self.group_name(start_group)
            if os.path.exists(group_path):
                #print(group_path)
                with open(group_path,'rb') as f:
                    obj=pickle.load(f)
                    yield obj
                start_group+=1

            else:
                break
            