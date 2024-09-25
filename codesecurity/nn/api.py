import torch
import os
import torch.utils.data as torchdata
from sklearn.metrics import confusion_matrix

def model_flatten_call(model:torch.nn.Module,input):
    if isinstance(input,list) or isinstance(input,tuple):
        return model(*input)
    
    return model(input)


def measure_train_model(model:torch.nn.Module,training_data:torchdata.DataLoader,test_data:torchdata.DataLoader,device,loss_func=torch.nn.CrossEntropyLoss(),model_call_handle=None,epoch_number=150,lr=1e-3,weight_decay=0,reinforce=None):
    if model_call_handle is None:
        model_call_handle=lambda model,x:model(x[0])

    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    
    status_per_epoch=[]
    

    for i in range(epoch_number):
        total_number=0
        print(f"epoch : {i}:")
        model.train()

        total_loss=0.
        total_accuracy=0

        for e in training_data:
            y=e[-1]
            x=e[:-1]
            if reinforce is not None:
                x=reinforce(x)
            y_hat=model_call_handle(model,x)
            #y_hat=torch.softmax(y_hat,1)
            if len(y.size())==1:
                y=torch.eye(y_hat.size(-1))[y]
            y=y.to(device)
            loss= loss_func(y_hat,y.double())
            total_loss+=loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():

                class_number=y_hat.size(-1)

                _,maxk=torch.topk(y_hat,min(class_number,5),dim=-1)


                total_accuracy+=(y.argmax(1).view(-1,1) == maxk[:,0:1]).sum().item()
                total_number+=y.size(0)
        print(f"train top1: {total_accuracy/total_number}")

        #model.eval()
        val_acc=eval_model(model,test_data,device,loss_func,model_call_handle)
        train_acc=total_accuracy/total_number
        
        status_per_epoch.append((total_loss,train_acc,val_acc))

    return status_per_epoch

def train_model(model:torch.nn.Module,training_data:torchdata.DataLoader,test_data:torchdata.DataLoader,device,loss_func=torch.nn.CrossEntropyLoss(),model_call_handle=None,out_file=None,epoch_number=150,lr=1e-3,weight_decay=0,reinforce=None):

    if model_call_handle is None:
        model_call_handle=lambda model,x:model(x[0])

    highest_accuracy=eval_model(model,test_data,device,loss_func,model_call_handle)

    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)

    for i in range(epoch_number):
        total_number=0
        print(f"epoch : {i}:")
        model.train()

        #total_test_loss=0
        total_accuracy=0

        for e in training_data:
            y=e[-1]
            x=e[:-1]
            if reinforce is not None:
                x=reinforce(x)
            y_hat=model_call_handle(model,x)
            #y_hat=torch.softmax(y_hat,1)
            if len(y.size())==1:
                y=torch.eye(y_hat.size(-1))[y]
            y=y.to(device)
            loss= loss_func(y_hat,y.double())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():

                class_number=y_hat.size(-1)

                _,maxk=torch.topk(y_hat,min(class_number,5),dim=-1)


                total_accuracy+=(y.argmax(1).view(-1,1) == maxk[:,0:1]).sum().item()
                total_number+=y.size(0)
        print(f"train top1: {total_accuracy/total_number}")

        #model.eval()
        now_accuary=eval_model(model,test_data,device,loss_func,model_call_handle)

        if out_file:
            if now_accuary>=highest_accuracy:
                torch.save(model.state_dict(),out_file)
                now_accuary=highest_accuracy

def validate_model(model:torch.nn.Module,data:torchdata.DataLoader,device,model_call_handle=None):
    if model_call_handle is None:
        model_call_handle=lambda model,x:model(x[0])
    model.eval()
    with torch.no_grad():
        for e in data:
            y=e[-1]
            y=y.to(device)
            y_hat=model_call_handle(model,e[:-1])

            y_hat=torch.softmax(y_hat,1)
            if len(y.size())==1:
                y=torch.eye(y_hat.size(-1),device=device)[y]

            yield y_hat,y
            
def eval_model(model:torch.nn.Module,data:torchdata.DataLoader,device,loss_func=torch.nn.CrossEntropyLoss(),model_call_handle=None):

    if model_call_handle is None:
        model_call_handle=lambda model,x:model(*x)

    total_number=0
    total_test_loss=0
    total_accuracy=0
    total_acc5=0
    number_of_iter=0
    class_number=0
    matrix=None
    model.eval()
    with torch.no_grad():
        for e in data:
            y=e[-1]
            y=y.to(device)
            y_hat=model_call_handle(model,e[:-1])

            if len(y.size())==1:
                y=torch.eye(y_hat.size(-1),device=device)[y]
            #print(y.size(),y_hat.size())

            #y_hat=torch.softmax(y_hat,1)

            loss= loss_func(y_hat,y.double())
            total_test_loss = total_test_loss + loss.item()

            class_number=y_hat.size(-1)

            _,maxk=torch.topk(y_hat,min(5,class_number),dim=-1)



            total_accuracy+=(y.argmax(1).view(-1,1) == maxk[:,0:1]).sum().item()
            total_acc5+=(y.argmax(1).view(-1,1)== maxk).sum().item()
            # y_hat_group=torch.stack([y_hat[i*test_dataset.group:(i+1)*test_dataset.group,:] for i in range(1)])
            # y_hat_group_eval=(torch.max(y_hat_group,2,keepdim=True)[0]==y_hat_group)

            # y_hat_group_eval=torch.sum(y_hat_group_eval,1)
            # y_group_eval=torch.stack([y[i*test_dataset.group] for i in range(1)])

            # group_acc=(y_hat_group_eval.argmax(1) == y_group_eval.argmax(1)).sum()
            # total_group_acc+=group_acc
            number_of_iter+=1
            total_number+=y.size(0)
            
            if class_number<=5:
                y_label=y.argmax(1).cpu().numpy()
                y_hat_label=y_hat.argmax(1).cpu().numpy()
                if matrix is None:
                    matrix=confusion_matrix(y_label,y_hat_label,labels=range(class_number))
                else:
                    matrix+=confusion_matrix(y_label,y_hat_label,labels=range(class_number))
        
        if class_number>5:
            print(f'loss :{total_test_loss/number_of_iter} top1:{total_accuracy/total_number} top5: {total_acc5/total_number}')
        else:
            print(f'loss :{total_test_loss/number_of_iter} top1:{total_accuracy/total_number}')
            num_classes = class_number
            class_accuracy = []
            for i in range(num_classes):
                class_correct = matrix[i, i]
                class_total = matrix[i, :].sum()
                accuracy = class_correct / class_total
                class_accuracy.append(accuracy)

            print(class_accuracy)
    
    return total_accuracy/total_number