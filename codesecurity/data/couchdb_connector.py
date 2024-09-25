import couchdb

class CouchDbConnector:
    def __init__(self, db_name, url):
        self.server=couchdb.Server(url)
        self.db=self.server[db_name]
        
    def get(self, id):
        return self.db.get(id)
    
    def ids(self):
        docs=self.doc_heads()
        ret=[doc.id for doc in docs]
        return ret
    
    def iter_docs(self,group=2000,max_times=None):
        start=0
        time=0
        while True:
            iterator=self.db.view('_all_docs',limit=group,skip=start)
            buffer=list(iterator)
                
            for row in buffer:
                yield self.get(row.id)
            
            start+=group
            if len(buffer)<group:
                break
            
            time+=1
            if max_times is not None and time>=max_times:
                break
            
    def topk_ids(self,k):
        
        iterator=self.db.view('_all_docs',limit=k)
        result=[]
        for e in iterator:
            result.append(e.id)
    
        return result
    
    def doc_heads(self):
        return list(self.db.view('_all_docs'))