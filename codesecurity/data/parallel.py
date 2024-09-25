import asyncio
import concurrent.futures
import shutil


def read_files_in_parallel(file_names,encoding='utf-8'):
    def read_file(file_name):
        with open(file_name, 'r',encoding=encoding) as file:
            content = file.read()
            return content
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交每个文件读取任务到线程池
        futures = [executor.submit(read_file, file_name) for file_name in file_names]
        
        # 获取每个任务的结果
        contents = []
        for future in concurrent.futures.as_completed(futures):
            try:
                content = future.result()
                contents.append(content)
            except Exception as e:
                print(f"An error occurred: {e}")
        
        return contents
    
def do_parallel(delegates,show_prcocess=False):
    
    total=len(delegates)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交每个文件读取任务到线程池
        futures = [executor.submit(handle,*args) for handle,args in delegates]
        
        # 获取每个任务的结果
        contents = []
        for future in concurrent.futures.as_completed(futures):
            try:
                content = future.result()
                contents.append(content)
                if show_prcocess:
                    print(f'processing : {len(contents)}/{total}')
            except Exception as e:
                contents.append(None)
                print(f"An error occurred: {e}")
        
        return contents

def do_parallel_lazy(delegates,parallel_number=10,show_prcocess=False):
    total=len(delegates)
    while len(delegates)>0:
        yield do_parallel(delegates[:min(parallel_number,len(delegates))],False)
        delegates=delegates[min(parallel_number,len(delegates)):]
        if show_prcocess:
            print(f'processing : {total-len(delegates)}/{total}')
            
async def copy_file(source, destination):
    try:
        shutil.copy2(source, destination)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

async def copy_files(file_pairs):
    tasks = []
    for source, destination in file_pairs:
        task = asyncio.create_task(copy_file(source, destination))
        tasks.append(task)
    
    # 等待所有任务完成
    await asyncio.gather(*tasks)

def copy_files_in_parallel(file_pairs):
    asyncio.run(copy_files(file_pairs))
    

if __name__ == '__main__':
    a=0
    def handle():
        global a
        a+=1
        return a
    
    pairs=[(handle,()) for i in range(10)]
    
    result=do_parallel(pairs)
    
    print(result)