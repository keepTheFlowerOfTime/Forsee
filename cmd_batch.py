import os
import sys

mode_list=['forsee_train','forsee_eval','forsee_robust_target',
           'forsee_robust_untarget','forse_validate_untarget','forse_validate_target','unuse_remove_test',
           'forsee_measure','magic']

mode='forsee_train'
argruments=[]

if len(sys.argv)>1:
    if sys.argv[1]=='m':
        mode=mode_list[-1]
    else:
        mode=sys.argv[1]

if mode=='forsee_train':
    cmds=['python main.py training_forsee dataset/gcj_cpp persp_cpp --k=8 --device=gpu',]
    #'python main.py training_forsee dataset/github_c github_c --k=10 --device=gpu --use_caches=False --rebuild={}',
    #'python main.py training_forsee dataset/java40 java40 --k=10 --device=gpu --use_caches=False --rebuild={}'

elif mode=='dataset_build':
    cmds=['python main.py build_training_caches dataset/gcj_cpp persp_cpp']

# if mode==mode_list[0]:
#     argruments=[['True']]
#     cmds=['python main.py training_forsee data/gcj_cpp gcj_cpp --k=8 --device=gpu --use_caches=False --rebuild={}',
#           'python main.py training_forsee data/a_github_c github_c --k=10 --device=gpu --use_caches=False --rebuild={}',
#           'python main.py training_forsee data/java40 java40 --k=10 --device=gpu --use_caches=False --rebuild={}']
#     #]

# elif mode==mode_list[1]:
#     cmds=[
#         'python main.py eval_forsee data/gcj_cpp gcj_cpp --k=8 --device=gpu',
#         'python main.py eval_forsee data/a_github_c github_c --k=10 --device=gpu',
#         'python main.py eval_forsee data/java40 java40 --k=10 --device=gpu'
#     ]

# elif mode==mode_list[2]:
#     argruments=[['True']]
#     function_name='eval_external_data_forsee'
#     # eval_external_data_forsee_vanille
#     if len(sys.argv)>=3:
#         if sys.argv[2]=='v':
#             function_name='eval_external_data_forsee_vanille'
    
#     cmds=[
#         f'python main.py {function_name} data/java40 generate_data/java40/program_file/targeted_attack_file model_caches/java40_target java40 --device=gpu --use_caches={{}}',
#         f'python main.py {function_name} data/gcj_cpp generate_data/gcj_cpp/program_file/targeted_attack_file model_caches/gcj_cpp_target gcj_cpp --device=gpu --use_caches={{}}',
#         f'python main.py {function_name} data/a_github_c generate_data/a_github_c/program_file/targeted_attack_file model_caches/a_github_c_target github_c --device=gpu --use_caches={{}}'
#     ]

# elif mode==mode_list[3]:
#     argruments=[['True']]
#     function_name='eval_external_data_forsee'
#     # eval_external_data_forsee_vanille
#     if len(sys.argv)>=3:
#         if sys.argv[2]=='v':
#             function_name='eval_external_data_forsee_vanille'
#     cmds=[
#         f'python main.py {function_name} data/java40 generate_data/java40/program_file/untargeted_attack_file model_caches/java40_untarget java40 --device=gpu --use_caches={{}}',
#         f'python main.py {function_name} data/gcj_cpp generate_data/gcj_cpp/program_file/untargeted_attack_file model_caches/gcj_cpp_untarget gcj_cpp --device=gpu --use_caches={{}}',
#         f'python main.py {function_name} data/a_github_c generate_data/a_github_c/program_file/untargeted_attack_file model_caches/a_github_c_untarget github_c --device=gpu --use_caches={{}}'
#     ]
# # 

# elif mode==mode_list[4]:
    
#     cmds=[
#         'python main.py validate_forsee  data/java40 java40 --target_caches_file=model_caches/java40_target/refine_data.pt --device=gpu --reverse=True',
#         'python main.py validate_forsee  data/gcj_cpp gcj_cpp --target_caches_file=model_caches/gcj_cpp_target/refine_data.pt --device=gpu --reverse=True',
#         'python main.py validate_forsee  data/a_github_c github_c --target_caches_file=model_caches/a_github_c_target/refine_data.pt --device=gpu --reverse=True'
#     ]

# elif mode==mode_list[5]:
#     cmds=[
#         'python main.py validate_forsee  data/java40 java40 --target_caches_file=model_caches/java40_untarget/refine_data.pt --device=gpu',
#         'python main.py validate_forsee  data/gcj_cpp gcj_cpp --target_caches_file=model_caches/gcj_cpp_untarget/refine_data.pt --device=gpu',
#         'python main.py validate_forsee  data/a_github_c github_c --target_caches_file=model_caches/a_github_c_untarget/refine_data.pt --device=gpu'
#     ]

# elif mode==mode_list[6]:
#     cmds=[
#         'python main.py training_forsee data/gcj_cpp_robust/train gcj_cpp --eval_dir=data/gcj_cpp_robust/validate --mode=std --device=gpu --rebuild=True',
#         'python main.py eval_external_data_forsee data/gcj_cpp_robust/train data/gcj_cpp_unuse_remove/validate model_caches/gcj_cpp_robust_validate gcj_cpp --device=gpu --use_caches=False'
#     ]
# elif mode==mode_list[7]:
#     cmds=['python main.py measurce_forsee data/gcj_cpp gcj_cpp --k=8 --device=gpu',
#           'python main.py measurce_forsee data/java40 java40 --k=10 --device=gpu',
#           'python main.py measurce_forsee data/a_github_c github_c --k=10 --device=gpu']


# elif mode==mode_list[-1]:
#     argruments=[['False']]
    
#     # eval_forsee_vanille training_forsee_vanille
#     function_name='eval_forsee_vanille'
    
#     dataset_lists=[]
#     dataset_lists.append(['data/gcj_cpp','data/a_github_c','data/java40'])
    
    
#     cmds=[f'python main.py {function_name} data/gcj_cpp gcj_cpp --k=8 --device=gpu --rebuild={{}}',
#           f'python main.py {function_name} data/a_github_c github_c --k=10 --device=gpu --rebuild={{}}',
#           f'python main.py {function_name} data/java40 java40 --k=10 --device=gpu --rebuild={{}}']

total=len(cmds)*max(len(argruments),1)       
i=1
for cmd in cmds:
    if len(argruments)==0:
        print(f'processing: {total}/{i}.')
        os.system(cmd)
        i+=1
    else:
        for argrument in argruments:
            print(f'processing: {total}/{i}. Use {argrument}')
            os.system(cmd.format(*argrument))
            i+=1


# python main.py training_forsee data/a_github_c github_c --k=10 --device=gpu --partial=101
# python main.py training_forsee data/java40 java40 --k=10 --device=gpu --partial=101