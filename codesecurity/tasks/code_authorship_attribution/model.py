import torch

from codesecurity.tasks.code_authorship_attribution.preprocessing import ForseeSuperParameter,DLCAISSuperParameter



class PreferenceEmbeding(torch.nn.Module):
    def __init__(self,super_parameters:ForseeSuperParameter,device) -> None:
        super().__init__()
        self.device=device
        
        self.layout_extractor=LayoutExtractor(super_parameters.layout_vector_dim,super_parameters.lay_hidden_dim,device,super_parameters.lay_channel)
        self.lexical_extractor=LexicalExtractor(super_parameters.lexical_vector_dim,super_parameters.lex_hidden_dim,device,super_parameters.lex_channel)
        self.syntactic_extractor=SyntacticExtractor(super_parameters.syntactic_vector_dim,super_parameters.syn_hidden_dim,device,super_parameters.syn_channel)
    
    @property
    def feature_dim(self):
        return self.layout_extractor.feature_dim+self.lexical_extractor.feature_dim+self.syntactic_extractor.feature_dim

    def forward(self,layout_x,lexical_y,syntactic_z):
        
        layout_x=layout_x.to(torch.float32).to(self.device)
        lexical_y=lexical_y.to(torch.float32).to(self.device)
        syntactic_z=syntactic_z.to(torch.float32).to(self.device)

        layout_x=self.layout_extractor(layout_x)
        lexical_y=self.lexical_extractor(lexical_y)
        syntactic_z=self.syntactic_extractor(syntactic_z)
        
        return layout_x,lexical_y,syntactic_z

class PreferenceModule(torch.nn.Module):
    def __init__(self,device) -> None:
        super().__init__()
        
        self.layout_w=torch.Tensor([1.]).to(device)
        self.lexical_w=torch.Tensor([1.]).to(device)
        self.syntactic_w=torch.Tensor([1.]).to(device)

        self.device=device

    def set_prefer(self,prefer_weight):
        if prefer_weight is None: return
        temp=torch.Tensor(prefer_weight).to(self.layout_w.device)

        self.layout_w[0]=temp[0]
        self.lexical_w[0]=temp[1]
        self.syntactic_w[0]=temp[2]

    def forward(self,layout_x,lexical_y,syntactical_z):
        layout_x=torch.flatten(layout_x,1)
        lexical_y=torch.flatten(lexical_y,1)
        syntactical_z=torch.flatten(syntactical_z,1)
        return torch.cat([self.layout_w*layout_x,self.lexical_w*lexical_y,self.syntactic_w*syntactical_z],1)

class PreferenceClassifier(torch.nn.Module):
    def __init__(self,input_dim,class_number,device) -> None:
        super().__init__()

        self.classifier=torch.nn.Linear(input_dim,class_number,bias=False,device=device)

    def forward(self,x):
        return self.classifier(x)

class PreferenceNetwork(torch.nn.Module):
   def __init__(self,embedding_module:PreferenceEmbeding,preference_module:PreferenceModule,classcifier:PreferenceClassifier) -> None: 
        super().__init__()
        self.embeding_module=embedding_module
        self.preference_module=preference_module
        self.classicifer=classcifier
   
   def embeding(self,layout_x,lexical_y,syntactic_z):
       x,y,z=self.embeding_module(layout_x,lexical_y,syntactic_z)

       style_vector=self.preference_module(x,y,z)
       
       return style_vector
   
   
   def forward(self,layout_x,lexical_y,syntactic_z):
        
        x,y,z=self.embeding_module(layout_x,lexical_y,syntactic_z)

        style_vector=self.preference_module(x,y,z)

        style_vector=torch.dropout(style_vector,0.5,self.training)

        result=self.classicifer(style_vector)

        return result


   def weights(self):
        return self.classicifer.classifier.weight

class PartialPreferenceNetwork(torch.nn.Module):
    def __init__(self,sp:ForseeSuperParameter,class_number,device,use_layout=True,use_lexical=True,use_syntactic=True) -> None: 
        super().__init__()
        self.extractors=[]
        if use_layout:
            self.extractors.append(LayoutExtractor(sp.layout_vector_dim,sp.lay_hidden_dim,device,sp.lay_channel))
        if use_lexical:
            self.extractors.append(LexicalExtractor(sp.lexical_vector_dim,sp.lex_hidden_dim,device,sp.lex_channel))
        if use_syntactic:
            self.extractors.append(SyntacticExtractor(sp.syntactic_vector_dim,sp.syn_hidden_dim,device,sp.syn_channel))
        #print(len(self.extractors))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        self.weights=[torch.Tensor([1.]).to(device) for i in range(len(self.extractors))]
        self.classicifer=torch.nn.Linear(self.feature_dim,class_number,bias=False,device=device)
        self.device=device
   
    def frozen_no_classicifer_parameters(self):
        for extractor in self.extractors:
            for param in extractor.parameters():
                param.requires_grad=False

        for weight in self.weights:
            weight.requires_grad=False
    
    @property
    def feature_dim(self):
        return sum([extractor.feature_dim for extractor in self.extractors])
    
    def forward(self,*args):
        assert len(args)==len(self.extractors)
        inputs=[input.to(torch.float32).to(self.device) for input in args]
        embedings=[extractor(input) for extractor,input in zip(self.extractors,inputs)]
        flatter_embeding=[torch.flatten(embeding,1) for embeding in embedings]

        style_vector=torch.cat([weight*embeding for weight,embeding in zip(self.weights,flatter_embeding)],1)

        style_vector=torch.dropout(style_vector,0.5,self.training)
        
        return self.classicifer(style_vector)
   
   
        
class LayoutExtractor(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,device,channel=8) -> None:
        super().__init__()
        
        self.lstm_layer_number=3
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.device=device
        self.channel=channel
        
        #self.linear_layer=torch.nn.Linear(1,hidden_dim,device=device)
        
    
        self.encoder=torch.nn.LSTM(1,self.hidden_dim,self.lstm_layer_number,batch_first=True,device=device)

        #self.encoder=self.conv_encoder()
        #self.encoder=torch.nn.TransformerEncoder(self.encoder_layer,3)

    @property
    def feature_dim(self):
        return self.hidden_dim
        
    def conv_encoder(self):

        def conv_block(in_channel,out_channel,kernel_size):
            block=torch.nn.Sequential(
                torch.nn.Conv1d(in_channel,in_channel,kernel_size,padding='same',device=self.device),
                torch.nn.Sigmoid(),
                torch.nn.Conv1d(in_channel,out_channel,kernel_size,padding='same',device=self.device),
                torch.nn.MaxPool1d(2),
                torch.nn.Sigmoid()
            )

            return block

        kernel_size=8
        start_channel=self.channel
        encoder=torch.nn.Sequential(
            torch.nn.Conv1d(1,start_channel,kernel_size,padding='same',device=self.device),
            torch.nn.Sigmoid(),
            conv_block(start_channel,2*start_channel,kernel_size),
            conv_block(2*start_channel,4*start_channel,kernel_size),
            #torch.nn.Conv1d(64,64,1,device=self.device),
            #torch.nn.Sigmoid()
        )

        return encoder
    
    def forward(self,x:torch.Tensor):
        
        assert len(x.shape)==2 
        
        
        x=x.to(self.device)

        x=x.reshape(-1,self.input_dim,1)

        #x=self.linear_layer(x)
        #x=torch.relu(x)



        #x=self.encoder(x)



        #x=torch.permute(x,[0,2,1])

        
        hx=torch.zeros(self.lstm_layer_number,x.size(0), self.hidden_dim).to(self.device)
        cx=torch.zeros(self.lstm_layer_number,x.size(0), self.hidden_dim).to(self.device)
        
        x,(hx,cx)=self.encoder(x,(hx,cx))

        x=torch.relu(x)

        x=x.sum(1)

        #x=torch.sigmoid(x)
        #print(x.shape)
        
        return x    

class LexicalExtractor(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,device,channel=8) -> None:
        super().__init__()
        
        self.lstm_layer_number=3
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.device=device
        self.channel=channel
        
        # Ax+b=> 输入10000维 Ax+b =>256维 =>  输出4维 1*10000**10000*256->1*256 256*4=>1*4 
        self.linear_layer=torch.nn.Linear(input_dim,hidden_dim,device=device)
        
        self.encoder=torch.nn.Sequential(
            torch.nn.Linear(1,channel,device=self.device),
            torch.nn.ReLU(),
            torch.nn.Linear(channel,channel,device=self.device)            
        )

        #self.linear_layer=torch.nn.Linear(1,hidden_dim,device=device)
        
        #self.encoder=torch.nn.LSTM(1,self.channel,self.lstm_layer_number,batch_first=True,device=device)

    @property
    def feature_dim(self):
        return self.hidden_dim    

    def forward(self,x:torch.Tensor):
        
        assert len(x.shape)==2 
        
        x=x.to(self.device)
        
        x=self.linear_layer(x)

        x=torch.relu(x)
        
        x=x.reshape(-1,self.hidden_dim,1)
        
        x=self.encoder(x)
        # hx=torch.zeros(self.lstm_layer_number,x.size(0), self.channel).to(self.device)
        # cx=torch.zeros(self.lstm_layer_number,x.size(0), self.channel).to(self.device)
        
        #x,(hx,cx)=self.encoder(x,(hx,cx))

        x=torch.relu(x)

        x=x.sum(2)
        
        return x         
    
class SyntacticExtractor(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,device,channel=8) -> None:
        super().__init__()
        
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.device=device
        self.channel=channel
        
        self.linear_layer=torch.nn.Linear(input_dim,hidden_dim,device=device)
        
        self.linear_module=torch.nn.Sequential(
            torch.nn.Linear(1,channel,device=self.device),
            torch.nn.ReLU(),
            torch.nn.Linear(channel,channel,device=self.device)            
        )
        
    @property
    def feature_dim(self):
        return self.hidden_dim    

    def forward(self,x:torch.Tensor):
        
        assert len(x.shape)==2 
        
        x=x.to(self.device)
        
        x=self.linear_layer(x)
        
        x=torch.relu(x)

        x=x.reshape(-1,self.hidden_dim,1)
        
        x=self.linear_module(x)

        x=torch.relu(x)

        x=x.sum(2)
        
        return x          
        
class IndependenceModel(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,mode,device,class_number,channel=8) -> None:
        super().__init__()
        
        extractor=None
        if mode in ['layout','lay']:
            extractor=LayoutExtractor(input_dim,hidden_dim,device,channel)

        elif mode in ['lexical','lex']:
            extractor=LexicalExtractor(input_dim,hidden_dim,device,channel)

        elif mode in ['syntactic','syntax','syn']:
            extractor=SyntacticExtractor(input_dim,hidden_dim,device,channel)

        # self.classifier=torch.nn.Sequential(
        #     torch.nn.Linear(extractor.feature_dim,extractor.feature_dim,device=device),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(extractor.feature_dim,class_number,device=device)
        # )

        self.device=device
        self.extractor=extractor
        
        #self.add_module(extractor.__str__(),extractor)

        self.classifier=torch.nn.Sequential(
            torch.nn.Linear(self.extractor.feature_dim,max(int(self.extractor.feature_dim*0.88),class_number),device=device),
            torch.nn.ReLU(),
            torch.nn.Linear(max(int(self.extractor.feature_dim*0.88),class_number),class_number,device=device)
        )
    
    def forward(self,x):
        
        assert len(x.shape)==2 
        
        x=x.to(torch.float32)
        x=x.to(self.device)

        x=self.extractor(x)

        #x=torch.flatten(x,1)
        x=torch.relu(x)

        x=torch.dropout(x,0.5,self.training)
        
        result=self.classifier(x)

        return result


class DLCAIS(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,device,class_number) -> None:
        super().__init__()   

        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.class_number=class_number
        self.lstm_layer_number=3
        self.device=device
        self.representation_dim=int(0.45*input_dim)
        #self.lstm=torch.nn.LSTM(input_dim,hidden_dim,self.lstm_layer_number,batch_first=True,device=device,dropout=0.6)
        self.lstm=[]
        #self.adapter=torch.nn.Linear(input_dim,int(input_dim*0.45),device=device)

        for i in range(self.lstm_layer_number):
            if i==0:
                self.lstm.append(torch.nn.LSTM(input_dim,hidden_dim,batch_first=True,device=device))
            else:
                self.lstm.append(torch.nn.LSTM(hidden_dim,hidden_dim,batch_first=True,device=device))
        # for name,param in self.encoder.named_parameters():
        #     if 'weight' in name:
        #         torch.nn.init.xavier_normal_(param)
        #     elif 'bias' in name:
        #         torch.nn.init.xavier_normal_(param,0)
        self.linear=torch.nn.Sequential(
            self.l2Linear(self.hidden_dim,self.representation_dim,device=device),
            torch.nn.ReLU(),
            self.l2Linear(self.representation_dim,self.representation_dim,device=device),
            torch.nn.ReLU(),
        )
        
        self.classifier=self.l2Linear(self.representation_dim,self.class_number,device=device)
    
    def l2Linear(self,input_dim,output_dim,device=None):
        if device is None:
            device=self.device
        layer=torch.nn.Linear(input_dim,output_dim,device=device)
        #torch.nn.init.xavier_normal_(layer.weight)
        #torch.nn.init.xavier_normal_(layer.bias,0)
        return layer
    
    def forward(self,x):
        x=self.embeding(x)

        #x=torch.sigmoid(x)
        #print(x.shape)
        
        return self.classifier(x)

    def embeding(self,x):
        assert len(x.shape)==2 
        
        x=x.to(torch.float32)
        x=x.to(self.device)

        #x=self.adapter(x)


        x=torch.unsqueeze(x,1)
        #x=x[:,:,0:self.input_dim]

        #x=self.linear_layer(x)
        #x=torch.relu(x)



        #x=self.encoder(x)

        #x=torch.permute(x,[0,2,1])
        for i in range(self.lstm_layer_number):
            hx=torch.randn(1,x.size(0), self.hidden_dim).to(self.device)
            cx=torch.randn(1,x.size(0),self.hidden_dim).to(self.device)
            x,(hx,cx)=self.lstm[i](x,(hx,cx))
            x=torch.dropout(x,0.6,self.training)
        
        x=self.linear(x[:,-1,:])
        
        return x
    
def prepare_forsee_model(sp:ForseeSuperParameter,device,class_number):
    embeding_module=PreferenceEmbeding(sp,device)
    preference_module=PreferenceModule(device)
    prference_classcifier=PreferenceClassifier(embeding_module.feature_dim,class_number,device)
    

    return PreferenceNetwork(embeding_module,preference_module,prference_classcifier)

def prepare_partial_forsee_model(sp:ForseeSuperParameter,device,class_number,use_layout=True,use_lexical=True,use_syntactic=True):

    return PartialPreferenceNetwork(sp,class_number,device,use_layout,use_lexical,use_syntactic)

def prepare_independence_model(sp:ForseeSuperParameter,device,class_number):
    layout_model=IndependenceModel(sp.layout_vector_dim,sp.lay_hidden_dim,'layout',device,class_number,sp.lay_channel)
    lexical_model=IndependenceModel(sp.lexical_vector_dim,sp.lex_hidden_dim,'lexical',device,class_number,sp.lex_channel)
    syntactic_model=IndependenceModel(sp.syntactic_vector_dim,sp.syn_hidden_dim,'syn',device,class_number,sp.syn_channel)
    
    return [layout_model,lexical_model,syntactic_model]

def prepare_select_independence_models(sp:ForseeSuperParameter,device,class_number,use_layout=True,use_lexical=True,use_syntactic=True):
    ret=[]

    if use_layout:
        ret.append(IndependenceModel(sp.layout_vector_dim,sp.lay_hidden_dim,'layout',device,class_number,sp.lay_channel))
    if use_lexical:
        ret.append(IndependenceModel(sp.lexical_vector_dim,sp.lex_hidden_dim,'lexical',device,class_number,sp.lex_channel))
    if use_syntactic:
        ret.append(IndependenceModel(sp.syntactic_vector_dim,sp.syn_hidden_dim,'syn',device,class_number,sp.syn_channel))
    return ret

def prepare_DLCAIS(sp:DLCAISSuperParameter,device,class_number):
    return DLCAIS(sp.input_dim,sp.hidden_dim,device,class_number)
    