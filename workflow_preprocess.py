'''
用来预处理XML文件, 获得各类属性参数: 任务集合、依赖关系、任务数量、任务执行时间、任务的前驱节点、
任务的后继节点、任务的入口节点、任务的出口节点、任务的ranku值
'''

from xml.etree.ElementTree import ElementTree
import numpy as np
import re, math
import xml.dom.minidom
from random import choice
from parameter import get_args


args = get_args()

# 处理科学工作流XML文件
class XMLtoDAG():
    adag_tag = "{http://pegasus.isi.edu/schema/DAX}adag"
    job_tag = "{http://pegasus.isi.edu/schema/DAX}job"
    child_tag = "{http://pegasus.isi.edu/schema/DAX}child"
    parent_tag = "{http://pegasus.isi.edu/schema/DAX}parent"
    uses_tag = "{http://pegasus.isi.edu/schema/DAX}uses"

    def __init__(self, file):
        self.xmlFile = file
        self.n_task = int(re.compile(r'\d+').findall(file)[0])
        self.dag = self.get_dag()  # 工作流的结构，表示为一个相邻的矩阵
        self.edge_num = self.get_edge_num()  # 工作流的边的数量
        self.edges = self.get_edge_data()  # 工作流的边集合(数据大小)
        self.task_type = np.zeros((self.n_task), dtype=int)  # 工作流中的任务类型
        self.runtime = self.get_runtime()  # 工作流中的任务执行时间
        self.inputfilesize = self.get_inputfilesize()  # 工作流中的任务的输入文件大小
        self.outputfilesize = self.get_outputfilesize()  # 工作流中的任务的输出文件大小
        self.privacy_security_level = self.get_privacy_security_level()  # 工作流中的任务的隐私安全级别
        self.precursor = self.get_precursor()  # 工作流中的任务的前驱节点
        self.successor = self.get_successor()  # 工作流中的任务的后继节点
        self.entry = self.find_entry()  # 工作流中的任务的入口节点
        self.exit = self.find_exit()  # 工作流中的任务的出口节点
        self.jobs = self.jobs() # 工作流中的任务集合
        
    # 使用minidom解释器获取工作流的dag
    def get_dag(self): 
        domtree = xml.dom.minidom.parse(self.xmlFile)
        collection = domtree.documentElement
        childrens = collection.getElementsByTagName("child")
        '''
        示例：
        <child ref="ID00000">
            <parent ref="ID00021"/>
            <parent ref="ID00012"/>
            <parent ref="ID00023"/>
            <parent ref="ID00010"/>
            <parent ref="ID00025"/>
            <parent ref="ID00006"/>
            <parent ref="ID00017"/>
            <parent ref="ID00027"/>
            <parent ref="ID00004"/>
            <parent ref="ID00015"/>
            <parent ref="ID00029"/>
            <parent ref="ID00008"/>
            <parent ref="ID00019"/>
        </child>
        '''
        dag = np.zeros((self.n_task, self.n_task))
        for child in childrens:
            child_id = child.getAttribute('ref') # child ref="ID00000"
            child_id = int(child_id[2:])  # ref ="00000" 
            # print('Child: ', child_id)
            parents = child.getElementsByTagName('parent') # 遍历每个子节点的父节点
            for parent in parents:
                parent_id = parent.getAttribute('ref') # parent ref="ID00021"
                parent_id = int(parent_id[2:])  # ref ="00021"
                # print(parent_id)
                dag[parent_id, child_id] = 1
        return dag # 返回工作流的dag(矩阵形式，有依赖关系的为1，无依赖关系的为0)

    # 获取工作流的前驱节点集合
    def get_precursor(self): 
        precursor = [ [] for i in range(self.n_task) ]
        dag = self.get_dag()
        for in_node in range(self.n_task):
            for out_node in range(self.n_task):
                if(dag[in_node][out_node] == 1):
                    precursor[out_node].append(in_node)
        return precursor

    # 获取工作流的后继节点集合
    def get_successor(self):    
        successor = [ [] for i in range(self.n_task) ]
        dag = self.get_dag()
        for in_node in range(self.n_task):
            for out_node in range(self.n_task):
                if(dag[in_node][out_node] != 0):
                    successor[in_node].append(out_node)
        return successor

    # 获取边(依赖关系)的集合
    def print_graph(self):  
        # print(self.dag)
        '''
        示例：[(2, 3), (2, 5), (2, 7), (2, 9), (2, 11), (3, 1), (3, 4), (4, 0), (5, 1), (5, 6), 
        (6, 0), (7, 1), (7, 8), (8, 0), (9, 1), (9, 10), (10, 0), (11, 1), (11, 12), (12, 0), 
        (13, 14), (13, 16), (13, 18), (13, 20), (13, 22), (13, 24), (13, 26), (13, 28), (14, 1),
          (14, 15), (15, 0), (16, 1), (16, 17), (17, 0), (18, 1), (18, 19), (19, 0), (20, 1), 
          (20, 21), (21, 0), (22, 1), (22, 23), (23, 0), (24, 1), (24, 25), (25, 0), (26, 1), 
          (26, 27), (27, 0), (28, 1), (28, 29), (29, 0)]
        '''
        edges = []
        for i in range(self.n_task):
            for j in range(self.n_task):
                if self.dag[i, j] != 0:
                    # print(i, ' -> ', j)
                    edge = (i, j)
                    edges.append(edge)
        # print(edges)
        return edges
    
    # 获取传输数据(对应边)大小(单位B)
    def get_edge_data(self):
        job_1 = [row for row in self.jobs() if row['id'] == 0]
        if job_1[0]['namespace'] == 'CyberShake':
            #with open('/home/user/hhy/PSW-DRL/XML_Scientific_Workflow/CyberShake/CyberShake_throughput.txt', 'r') as file:
            with open('XML_Scientific_Workflow\CyberShake\CyberShake_throughput.txt', 'r') as file: # # 打开文件
                text = file.read() # # 读取文件内容
            groups = text.split('\n\n') # 利用空行将连续的数字分组
            number_arrays = [] # 创建存储数字组的列表
            edges = np.zeros((self.n_task, self.n_task))
            # 遍历每个分组，将数字存储到数组中
            for group in groups:
                numbers = group.strip().split(',')
                number_array = [int(num) for num in numbers]
                number_arrays.append(number_array)
            if self.n_task == 30:
                edge = number_arrays[0]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1         
                return edges
            elif self.n_task == 50:
                edge = number_arrays[1]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges
            elif self.n_task == 100:
                edge = number_arrays[2]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges
            elif self.n_task == 1000:
                edge = number_arrays[3]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges
        
        elif job_1[0]['namespace'] == 'Genome':
            #with open('/home/user/hhy/PSW-DRL/XML_Scientific_Workflow/Epigenomics/Epigenomics_throughput.txt', 'r') as file:
            with open('XML_Scientific_Workflow\Epigenomics\Epigenomics_throughput.txt', 'r') as file: # # 打开文件
                text = file.read() # # 读取文件内容
            groups = text.split('\n\n') # 利用空行将连续的数字分组
            number_arrays = [] # 创建存储数字组的列表
            edges = np.zeros((self.n_task, self.n_task))
            # 遍历每个分组，将数字存储到数组中
            for group in groups:
                numbers = group.strip().split(',')
                number_array = [int(num) for num in numbers]
                number_arrays.append(number_array)
            if self.n_task == 24:
                edge = number_arrays[0]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges
            elif self.n_task == 47:
                edge = number_arrays[1]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges
            elif self.n_task == 100:
                edge = number_arrays[2]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges
            elif self.n_task == 997:
                edge = number_arrays[3]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges
            
        elif job_1[0]['namespace'] == 'LIGO':
            #with open('/home/user/hhy/PSW-DRL/XML_Scientific_Workflow/LIGO/Inspiral_throughput.txt', 'r') as file:
            with open('XML_Scientific_Workflow\LIGO\Inspiral_throughput.txt', 'r') as file:
                text = file.read()
            groups = text.split('\n\n')
            number_arrays = [] 
            edges = np.zeros((self.n_task, self.n_task))
            for group in groups:
                numbers = group.strip().split(',')
                number_array = [int(num) for num in numbers]
                number_arrays.append(number_array)
            if self.n_task == 30:
                edge = number_arrays[0]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges
            elif self.n_task == 50:
                edge = number_arrays[1]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges
            elif self.n_task == 100:
                edge = number_arrays[2]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges
            elif self.n_task == 1000:
                edge = number_arrays[3]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges
        
        elif job_1[0]['namespace'] == 'Montage':
            #with open('/home/user/hhy/PSW-DRL/XML_Scientific_Workflow/Montage/Montage_throughput.txt', 'r') as file:
            with open('XML_Scientific_Workflow\Montage\Montage_throughput.txt', 'r') as file:
                text = file.read()
            groups = text.split('\n\n')
            number_arrays = [] 
            edges = np.zeros((self.n_task, self.n_task))
            for group in groups:
                numbers = group.strip().split(',')
                number_array = [int(num) for num in numbers]
                number_arrays.append(number_array)
            if self.n_task == 25:
                edge = number_arrays[0]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges
            elif self.n_task == 50:
                edge = number_arrays[1]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges
            elif self.n_task == 100:
                edge = number_arrays[2]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges
            elif self.n_task == 1000:
                edge = number_arrays[3]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges
            
        elif job_1[0]['namespace'] == 'SIPHT':
            #with open('/home/user/hhy/PSW-DRL/XML_Scientific_Workflow/SIPHT/Sipht_throughput.txt', 'r') as file:
            with open('XML_Scientific_Workflow\SIPHT\Sipht_throughput.txt', 'r') as file:
                text = file.read()
            groups = text.split('\n\n')
            number_arrays = [] 
            edges = np.zeros((self.n_task, self.n_task))
            for group in groups:
                numbers = group.strip().split(',')
                number_array = [int(num) for num in numbers]
                number_arrays.append(number_array)
            if self.n_task == 29:
                edge = number_arrays[0]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges
            elif self.n_task == 58:
                edge = number_arrays[1]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges
            elif self.n_task == 97:
                edge = number_arrays[2]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges
            elif self.n_task == 968:
                edge = number_arrays[3]
                count = 0
                for i in range(self.n_task):
                    for j in range(self.n_task):
                        if self.dag[i, j] != 0:
                            edges[i, j] = edge[count]
                            count += 1 
                return edges

    # 获取边的数量
    def get_edge_num(self):
        edges_num = 0
        for i in range(self.n_task):
            for j in range(self.n_task):
                if self.dag[i, j] != 0:
                    edges_num += 1
        return edges_num

    # 工作流中的任务集合
    def jobs(self):
        """
        任务属性: id, name(task type), namespace(workflow), runtime, size(file)
        id : 任务的编号
        namespace : 任务所属工作流
        name : 任务类型
        runtime : 任务长度
        file : 任务的输入输出文件, size为文件大小, link属性表示文件的种类(input输入文件或output输出文件)
        """

        tree = ElementTree(file=self.xmlFile)
        root = tree.getroot()
        simple_jobs = []
        pattern = re.compile(r'\+?[1-9][0-9]*$|0$')    # 匹配第一个不为0的数字或者以0结尾的数字
        imagetypes = [200*1024*1024, 400*1024*1024, 1000*1024*1024, 2000*1024*1024]             # 镜像文件的大小， unit: *B
        for job in root.iter(tag=self.job_tag):
            input_size = []
            # print(len(job.findall(self.uses_tag)))
            for use in job.findall(self.uses_tag):
                if use.get('link')=='input':
                    use_input_file_size = int(use.get('size'))    # the usage files' size (unit: B)
                    # input_speed.append(round(use_file_size/(10*1024*1024),6))    # Network I/O bandwidth=10M/s
                    input_size.append(use_input_file_size)
                if use.get('link')=='output':
                    output_size = int(use.get('size'))
    
            Job_Privacy_Factor = args.privacy_factor

            simple_job = {'id': int(pattern.findall(job.attrib['id'])[0]), 
                            'name': job.attrib['name'], 
                            'namespace': job.attrib['namespace'],
                            'runtime': float(job.attrib['runtime']),    
                            'inputfilesize': sum(input_size),    # the total size of a job:  * B
                            'outputfilesize': output_size,      # the output sie of a job: *B
                            'imagesize': choice(imagetypes),   # the image size of the containerized task
                            # 加入隐私和安全参数，随机取值（根据分级）
                            # 判断是否为入口/出口节点
                            'privacy_security_level': np.random.choice([1,2,3], p=[Job_Privacy_Factor,(1-Job_Privacy_Factor)/2, (1-Job_Privacy_Factor)/2])
                            #'privacy_security_level': random.randint(1,3),    # 隐私安全模型等级
                            
                          }
            simple_jobs.append(simple_job)
        '''
        示例：
        [{'id': 0, 'name': 'ZipPSA', 'namespace': 'CyberShake', 'runtime': 0.06, 'inputfilesize': 2808, 'outputfilesize': 172, 'imagesize': 2097152000}, 
        {'id': 1, 'name': 'ZipSeis', 'namespace': 'CyberShake', 'runtime': 0.08, 'inputfilesize': 312000, 'outputfilesize': 18657, 'imagesize': 1048576000},
        '''
        return simple_jobs
    
    # 根据任务的编号ID返回任务属性
    def get_node(self, id):
        for job in self.jobs():
            # print(job)
            if id==job['id']:
                return job
            
    # 工作流中的任务类型的集合
    def types(self):  
        types = []
        res = []
        for job in self.jobs(): 
            types.append(job['name'])
        for i, type in enumerate({}.fromkeys(types).keys()):
            res.append(type)
        for i, type in enumerate(types):
            self.task_type[i]=res.index(type)
            # print(self.taskType[i])
        return res, self.task_type
    
    # 各类型任务的runtime(任务长度)集合
    def typeRTimeDicts(self, types, jobs):  
        typeRTimeDict = {}
        for typ in types:
            lst = []
            for job in jobs:
                if job['name'] == typ:
                    lst.append(job['runtime'])
            print(typ, lst)
            typeRTimeDict[typ] = lst
        return typeRTimeDict
    
    # 各类型任务的transfer time集合
    def typeTTimeDicts(self, types, jobs):  # the set of transfer time for each type of jobs
        typTTimeDict = {}
        for typ in types:
            lst = []
            for job in jobs:
                if job['name'] == typ:
                    lst.append(job['transtime'])
            typTTimeDict[typ] = lst
        return typTTimeDict
    
    # 通信时间二维矩阵（即edge）
    def transtime(self):
        transtime = np.zeros((self.n_task, self.n_task))
        for i in range(self.n_task):
            for j in range(self.n_task):
                if self.dag[i, j] != 0:
                    #transtime[i, j] = self.get_node(i)['transtime']
                    node = self.get_node(i)
                    # transtime[i, j] = [job['transtime'] for job in self.jobs() if job['id'] == node][0]
                    transtime[i, j] = 0.2
        return transtime

    # 寻找入口节点
    def find_entry(self):
        entry = [0] * self.n_task
        precursor = self.get_precursor()
        for i in range(self.n_task):
            if(len(precursor[i]) == 0):
                    entry[i] = 1
        return entry

    # 寻找出口节点
    def find_exit(self):
        exit = [0] * self.n_task
        successor = self.get_successor()
        for i in range(self.n_task):
            if(len(successor[i]) == 0):
                    exit[i] = 1
        return exit

    # 添加虚拟入口节点
    def add_virtual_entry(self):
    #def add_virtual_entry(self,jobs):
        virtual_entry = {'id': -1, 'name': 'virtual_entry', 'namespace': 'virtual_entry', 'runtime': 0, 'inputfilesize': 0, 'outputfilesize': 0, 'imagesize': 0}
        self.jobs.insert(0, virtual_entry)
        self.precursor.insert(0, [])
        self.successor.insert(0, [])
        for i in range(self.n_task):
            # 如果此节点没有前驱节点则设置虚拟入口为其前驱节点
            if(self.entry[i] == 1):
                self.precursor[i+1].append(virtual_entry['id'])
                self.successor[0].append(i)
            # 还需要加入input/output文件的大小(添加至连接虚拟入口/出口的节点)
        #self.jobs().append(virtual_entry)

    # 添加虚拟出口节点
    def add_virtual_exit(self):
        virtual_exit = {'id': -2, 'name': 'virtual_entry', 'namespace': 'virtual_entry', 'runtime': 0, 'inputfilesize': 0, 'outputfilesize': 0, 'imagesize': 0}
        self.jobs.append(virtual_exit)
        self.precursor.append([])
        self.successor.append([])
        for i in range(self.n_task):
            # 如果此节点没有后继节点则设置虚拟出口为其后继节点
            if(self.exit[i] == 1):
                self.successor[i+1].append(virtual_exit['id'])
                self.precursor[self.n_task + 1].append(i)
            # 还需要加入input/output文件的大小(添加至连接虚拟入口/出口的节点)

    # 返回虚拟入口节点任务的编号  (就是-1吧）
    def virtual_entry_index(self):
        for job in self.jobs():
            if job['name'] == 'virtual_entry':
                index = job['id']
                #return job['id']
                return index

    # 用来返回参数，后面模型需要什么就返回什么
    def dag(self):
        n_task = self.n_task
        runtime = [row['runtime'] for row in self.jobs()]
        transtime = self.transtime()
        precursor = self.get_precursor()
        successor = self.get_successor()
        entry = self.find_entry()
        exit = self.find_exit()
        
        return n_task, runtime, transtime, precursor, successor, entry, exit

    # 返回任务的runtime
    def get_runtime(self):
        runtime = [row['runtime'] for row in self.jobs()]
        return runtime
    
    # 返回任务的privacy_security_level
    def get_privacy_security_level(self):
        privacy_security_level = [row['privacy_security_level'] for row in self.jobs()]
        return privacy_security_level

    # 返回任务的inputfilesize
    def get_inputfilesize(self):
        inputfilesize = [row['inputfilesize'] for row in self.jobs()]
        return inputfilesize
    
    # 返回任务的outputfilesize
    def get_outputfilesize(self):
        outputfilesize = [row['outputfilesize'] for row in self.jobs()]
        return outputfilesize
    

    
