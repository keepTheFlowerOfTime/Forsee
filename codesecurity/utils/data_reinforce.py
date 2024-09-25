from typing import Any
import torch
import torchvision.transforms as transforms
import numpy as np
class ImageReinforcer:
    def __init__(self,transform_list:list):
        self.transform_list=transform_list
    
    def __call__(self, img) -> Any:
        return self.transform(img)
    
    def transform(self,img):
        for t in self.transform_list:
            img=t(img)
        return img
    
    @staticmethod
    def default():
        # 随机裁剪 randomcrop
        transform_list=[]
        transform_list.append(RandomHorizonCrop())
        
        obj=ImageReinforcer(transform_list)
        
        return obj

class RandomHorizonCrop:
    def __init__(self,dev=2) -> None:
        self.dev=dev
    
    def _crop(self,img:torch.Tensor):
        y=np.random.randint(1,self.dev)
        return img[:,y:,],y
    
    def __call__(self,img:torch.Tensor):
        ret=torch.zeros_like(img)
        
        
        if len(img.shape)==4:
            for i in range(img.shape[0]):
                croped,y=self._crop(img[i])
                ret[i,:,:-y,:]=croped
        
        if len(img.shape)==3:
            croped,y=self._crop(img)
            ret[:,:-y,:]=croped
        
        return ret

    
if __name__=="__main__":
    instance=ImageReinforcer.default()
    
    img=torch.rand(4,4,2)
    print(img.permute(2,0,1))
    
    print(instance.transform(img).permute(2,0,1))



