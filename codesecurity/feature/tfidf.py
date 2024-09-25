
import pickle

import gensim


import numpy as np
import string
from functools import partial

from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer


def good_token(token:str):
    allow_start=string.ascii_letters+'_'
    
    if len(token)==0: return False
    
    if token[0] in allow_start: return True
    
    return False

class TfidfModule:

    @staticmethod
    def empty_tokenizer(x):
        return x

    def __init__(self) -> None:
        self.vocab2id=CountVectorizer(lowercase=False,tokenizer=TfidfModule.empty_tokenizer)
        self.tfidf=TfidfTransformer()
        self.docs=[]
        
    def add_documents(self,docs):
        if isinstance(docs[0],str):
            docs=[docs]

        self.docs+=docs
        
        #print(self.vocab2id.vocabulary_.__len__())
        #print(bows.shape)

        #print(self.tf.shape)

        #self.vocab2id.add_documents(docs,prune_at=100000)
        #self.doc_tfs+=[self.vocab2id.doc2bow(x) for x in docs]
    
    def build(self):
        
        # tokens=self.vocab2id.token2id.keys()
        # tokens=[x for x in tokens if good_token(x)]
        # ids=[self.vocab2id.token2id[x] for x in tokens]
        # self.vocab2id.filter_tokens(good_ids=ids)
        #self.tf=np.array(self.tf)[0]

        bows=self.vocab2id.fit_transform(self.docs)
        self.tfidf.fit(bows)
        self.tf=np.sum(bows.toarray(),axis=0)
        self.select_locations=np.argsort(-self.tf)

        del self.docs
        #print(self.select_locations)
        #print(self.select_locations)
        #self.tfidf=TfidfTransformer()

        #self.tfidf=gensim.models.TfidfModel(dictionary=self.vocab2id,wglobal=partial(gensim.models.tfidfmodel.df2idf,add=1.0),normalize=False)
        return self.tfidf
    
    def save(self,out_file):
        if out_file:
            with open(out_file,'wb') as f:
                pickle.dump(self,f)

    def dev_tfidf(self,bow):
        total=sum([e[1] for e in bow])
        termid_array, tf_array = [], []
        for termid, tf in bow:
            termid_array.append(termid)
            tf_array.append(tf/total)

        vector = [
            (termid, tf * (self.tfidf.idfs.get(termid,0.0)+1))
            for termid, tf in zip(termid_array, tf_array)
        ]
        
        return vector
    
    def compute_tfidf_(self,docs):
        if isinstance(docs[0],str):
            bow=self.vocab2id.doc2bow(docs)
            return self.dev_tfidf(bow)
        
        else:
            bows=[self.vocab2id.doc2bow(x) for x in docs]
            return [self.dev_tfidf(x) for x in bows]

    def compute_tfidf(self,docs):
        if isinstance(docs[0],str):
            docs=[docs]
            bows=self.vocab2id.transform(docs)
            return self.tfidf.transform(bows).toarray()[0]
            #return self.tfidf[bow]
        
        else:
            bows=self.vocab2id.transform(docs)
            return self.tfidf.transform(bows).toarray()
    
    def get_tfidf_vec(self,tokens,vec_dim):
        

        tfidf_vec=self.compute_tfidf(tokens)

        #print(tfidf_vec)

        new_vec_dim=min(vec_dim,self.select_locations.shape[0])
        

        if len(tfidf_vec.shape)==1:
            ret=tfidf_vec[self.select_locations[:new_vec_dim]]
            if self.select_locations.shape[0]<vec_dim:
                ret=np.concatenate([ret,np.zeros([vec_dim-ret.shape[0]])])
            return ret
        
        else:
            ret=[tfidf_vec[i][self.select_locations[:new_vec_dim]] for i in range(tfidf_vec.shape[0])]
            if self.select_locations.shape[0]<vec_dim:
                ret=[np.concatenate([x,np.zeros([vec_dim-x.shape[0]])]) for x in ret]
            return ret
    
    
    @staticmethod
    def load(path):
        with open(path,'rb') as f:
            return pickle.load(f)