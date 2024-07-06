'''
用来选择工作流workflow
'''

"""
import numpy as np
from workflow_preprocess import XMLtoDAG
from pathlib import Path

# 选择科学工作流
class Scientific_Workflow:
    def __init__(self, name, n_jobs):
        self.name = name
        self.n_jobs = n_jobs
    
    def get_workflow(self):
        CyberShake_folder = Path("XML_Scientific_Workflow/CyberShake/")
        CyberShake_30 = XMLtoDAG(open(CyberShake_folder / "CyberShake_30.xml"))
        CyberShake_50 = XMLtoDAG(CyberShake_folder / "CyberShake_50.xml")
        CyberShake_100 = XMLtoDAG(CyberShake_folder / "CyberShake_100.xml")
        CyberShake_1000 = XMLtoDAG(CyberShake_folder / "CyberShake_1000.xml")
        
        Epigenomics_folder = Path("XML_Scientific_Workflow/Epigenomics/")
        Epigenomics_24 = XMLtoDAG(Epigenomics_folder / "Epigenomics_24.xml")
        Epigenomics_47 = XMLtoDAG(Epigenomics_folder / "Epigenomics_47.xml")
        Epigenomics_100 = XMLtoDAG(Epigenomics_folder / "Epigenomics_100.xml") 
        Epigenomics_997 = XMLtoDAG(Epigenomics_folder / "Epigenomics_997.xml")

        Inspiral_folder = Path("XML_Scientific_Workflow/LIGO/")
        Inspiral_30 = XMLtoDAG(Inspiral_folder / "Inspiral_30.xml")
        Inspiral_50 = XMLtoDAG(Inspiral_folder / "Inspiral_50.xml")
        Inspiral_100 = XMLtoDAG(Inspiral_folder / "Inspiral_100.xml")
        Inspiral_1000 = XMLtoDAG(Inspiral_folder / "Inspiral_1000.xml")

        Montage_folder = Path("XML_Scientific_Workflow/Montage/")
        Montage_25 = XMLtoDAG(open(Montage_folder / "Montage_25.xml"))
        Montage_50 = XMLtoDAG(Montage_folder / "Montage_50.xml")
        Montage_100 = XMLtoDAG(Montage_folder / "Montage_100.xml")
        Montage_1000 = XMLtoDAG(Montage_folder / "Montage_1000.xml")

        Sipht_folder = Path("XML_Scientific_Workflow/SIPHT/")
        Sipht_29 = XMLtoDAG(Sipht_folder / "Sipht_29.xml")
        Sipht_58 = XMLtoDAG(Sipht_folder / "Sipht_58.xml")
        Sipht_97 = XMLtoDAG(Sipht_folder / "Sipht_97.xml")
        Sipht_968 = XMLtoDAG(Sipht_folder / "Sipht_968.xml")
        

        if self.name == 'CyberShake':
            if self.n_jobs == 30:
                return CyberShake_30
            if self.n_jobs == 50:
                return CyberShake_50
            if self.n_jobs == 100:
                return CyberShake_100
            if self.n_jobs == 1000:
                return CyberShake_1000
        
        if self.name == 'Epigenomics':
            if self.n_jobs == 24:
                return Epigenomics_24
            if self.n_jobs == 47:
                return Epigenomics_47
            if self.n_jobs == 100:
                return Epigenomics_100
            if self.n_jobs == 997:
                return Epigenomics_997
        

        if self.name == 'Inspiral':
            if self.n_jobs == 30:
                return Inspiral_30
            if self.n_jobs == 50:
                return Inspiral_50
            if self.n_jobs == 100:
                return Inspiral_100
            if self.n_jobs == 1000:
                return Inspiral_1000
        
        if self.name == 'Montage':
            if self.n_jobs == 25:
                return Montage_25
            if self.n_jobs == 50:
                return Montage_50
            if self.n_jobs == 100:
                return Montage_100
            if self.n_jobs == 1000:
                return Montage_1000
        
        if self.name == 'Sipht':
            if self.n_jobs == 29:
                return Sipht_29
            if self.n_jobs == 58:
                return Sipht_58
            if self.n_jobs == 97:
                return Sipht_97
            if self.n_jobs == 968:
                return Sipht_968
"""

'''
用来选择工作流workflow
'''
import numpy as np
from workflow_preprocess import XMLtoDAG

# 选择科学工作流
class Scientific_Workflow:
    def __init__(self, name, n_jobs):
        self.name = name
        self.n_jobs = n_jobs
    
    def get_workflow(self):      
        CyberShake_30 = XMLtoDAG('XML_Scientific_Workflow\CyberShake\CyberShake_30.xml') 
        CyberShake_50 = XMLtoDAG('XML_Scientific_Workflow\CyberShake\CyberShake_50.xml') 
        CyberShake_100 = XMLtoDAG('XML_Scientific_Workflow\CyberShake\CyberShake_100.xml') 
        CyberShake_1000 = XMLtoDAG('XML_Scientific_Workflow\CyberShake\CyberShake_1000.xml') 
        

        Epigenomics_24 = XMLtoDAG('XML_Scientific_Workflow\Epigenomics\Epigenomics_24.xml') 
        Epigenomics_47 = XMLtoDAG('XML_Scientific_Workflow\Epigenomics\Epigenomics_47.xml') 
        Epigenomics_100 = XMLtoDAG('XML_Scientific_Workflow\Epigenomics\Epigenomics_100.xml') 
        Epigenomics_997 = XMLtoDAG('XML_Scientific_Workflow\Epigenomics\Epigenomics_997.xml') 

        Inspiral_30 = XMLtoDAG('XML_Scientific_Workflow\LIGO\Inspiral_30.xml') 
        Inspiral_50 = XMLtoDAG('XML_Scientific_Workflow\LIGO\Inspiral_50.xml')
        Inspiral_100 = XMLtoDAG('XML_Scientific_Workflow\LIGO\Inspiral_100.xml')
        Inspiral_1000 = XMLtoDAG('XML_Scientific_Workflow\LIGO\Inspiral_1000.xml')

        Montage_25 = XMLtoDAG('XML_Scientific_Workflow\Montage\Montage_25.xml')
        Montage_50 = XMLtoDAG('XML_Scientific_Workflow\Montage\Montage_50.xml')
        Montage_100 = XMLtoDAG('XML_Scientific_Workflow\Montage\Montage_100.xml')
        Montage_1000 = XMLtoDAG('XML_Scientific_Workflow\Montage\Montage_1000.xml')

        Sipht_29 = XMLtoDAG('XML_Scientific_Workflow\SIPHT\Sipht_29.xml')
        Sipht_58 = XMLtoDAG('XML_Scientific_Workflow\SIPHT\Sipht_58.xml')
        Sipht_97 = XMLtoDAG('XML_Scientific_Workflow\SIPHT\Sipht_97.xml')
        Sipht_968 = XMLtoDAG('XML_Scientific_Workflow\SIPHT\Sipht_968.xml')
        

        if self.name == 'CyberShake':
            if self.n_jobs == 30:
                return CyberShake_30
            if self.n_jobs == 50:
                return CyberShake_50
            if self.n_jobs == 100:
                return CyberShake_100
            if self.n_jobs == 1000:
                return CyberShake_1000
        
        if self.name == 'Epigenomics':
            if self.n_jobs == 24:
                return Epigenomics_24
            if self.n_jobs == 47:
                return Epigenomics_47
            if self.n_jobs == 100:
                return Epigenomics_100
            if self.n_jobs == 997:
                return Epigenomics_997
        

        if self.name == 'Inspiral':
            if self.n_jobs == 30:
                return Inspiral_30
            if self.n_jobs == 50:
                return Inspiral_50
            if self.n_jobs == 100:
                return Inspiral_100
            if self.n_jobs == 1000:
                return Inspiral_1000
        
        if self.name == 'Montage':
            if self.n_jobs == 25:
                return Montage_25
            if self.n_jobs == 50:
                return Montage_50
            if self.n_jobs == 100:
                return Montage_100
            if self.n_jobs == 1000:
                return Montage_1000
        
        if self.name == 'Sipht':
            if self.n_jobs == 29:
                return Sipht_29
            if self.n_jobs == 58:
                return Sipht_58
            if self.n_jobs == 97:
                return Sipht_97
            if self.n_jobs == 968:
                return Sipht_968
        