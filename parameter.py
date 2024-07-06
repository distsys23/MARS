"""
Code for parameters of VMs in hybrid cloud and the training process.
"""

import argparse

def parameter_parser():
    parser = argparse.ArgumentParser(description="SAIRL")

    """
    add_argument()方法

        name or flags - 一个命名或者一个选项字符串的列表
        
        action - 表示该选项要执行的操作
        
        default - 当参数未在命令行中出现时使用的值
        
        dest - 用来指定参数的位置
        
        type - 为参数类型,例如int
        
        choices - 用来选择输入参数的范围。例如choice = [1, 5, 10], 表示输入参数只能为1,5 或10
        
        help - 用来描述这个选项的作用
    """
    # number of tasks
    parser.add_argument("--task_num",
                        type=int,
                        default=968,
                        help="task num")

    '''
    parameters of VMs
    '''
    # number of VMs
    parser.add_argument("--VM_num",
                        type=int,
                        default=16,
                        help="VM num")
    # number of private cloud VMs
    parser.add_argument("--private_cloud_num",
                        type=int,
                        default=8,
                        help="private cloud num")
    # number of public cloud VMs
    parser.add_argument("--public_cloud_num",
                        type=int,
                        default=8,
                        help="public cloud num")
    # price of VMs
    
    parser.add_argument("--vm_cost",
                        type=list,
                        default=[0.024, 0.039, 0.056, 0.075, 0.096, 0.122, 0.151, 0.189, 0.12, 0.195, 0.28, 0.375, 0.48, 0.61, 0.755, 0.945],
                        help="vm cost")
    '''
    parser.add_argument("--vm_cost",
                        type=list,
                        default=[0.12, 0.195, 0.28, 0.375, 0.48, 0.61, 0.755, 0.945],
                        help="vm cost")
    '''
    # price of private cloud VMs
    parser.add_argument("--private_cloud_cost",
                        type=list,
                        default=[0.024, 0.039, 0.056, 0.075, 0.096, 0.122, 0.151, 0.189],
                        help="private cloud cost")
    # price of public cloud VMs
    parser.add_argument("--public_cloud_cost",
                        type=list,
                        default=[0.12, 0.195, 0.28, 0.375, 0.48, 0.61, 0.755, 0.945],
                        help="public cloud cost")
    # capacity of VMs
    parser.add_argument("--vm_capacity",
                        type=list,
                        default=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
                        help="vm capacity")
    # capacity of private cloud VMs
    parser.add_argument("--private_cloud_capacity",
                        type=list,
                        default=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
                        help="private cloud capacity")
    # capacity of public cloud VMs
    parser.add_argument("--public_cloud_capacity",
                        type=list,
                        default=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
                        help="public cloud capacity")
    # speed of VMs
    
    parser.add_argument("--vm_speed",
                        type=list,
                        # default=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
                        default=[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.5],
                        help="vm speed")
    '''
    parser.add_argument("--vm_speed",
                        type=list,
                        default=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
                        help="vm speed")
    '''
    # speed of private cloud VMs
    parser.add_argument("--private_cloud_speed",
                        type=list,
                        default=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
                        help="private cloud speed")
    # speed of public cloud VMs
    parser.add_argument("--public_cloud_speed",
                        type=list,
                        default=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5],
                        help="Public Cloud VM speed")
    # type of VMs
    parser.add_argument("--vm_type",
                        type=list,
                        #default=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                        default=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                        help="VM type")
    # privacy factor
    parser.add_argument("--privacy_factor",
                        type=int,
                        default=0.5,
                        help="Privacy factor of each jobs")
    # security
    parser.add_argument("--vm_security",
                        type=list,
                        # default=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
                        #default=[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        default=[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                        help="VM type")

    return parser.parse_args()


# get parameters
def get_args():
    args = parameter_parser()
    return args