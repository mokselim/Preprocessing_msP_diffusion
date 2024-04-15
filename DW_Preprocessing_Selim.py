#!/usr/bin/env python
# coding: utf-8
#Date: September 2023
#@Script by Mohamed Selim to run DWI preprocessing and DTI fitting, 
#also, it runs registration of the extracted FA map to a chosen Template, then applying the transformation to the extracted MD, RD and AD maps

#Prerequisites:
#1-Acquisition scheme 43 Grad directions (3 B0 + 20 B1000 + 20 B2500) 
#2-Nipype installed and its dependencies.
#3-Data converted to Nifti and augmented. 



#1
#Importing Libraries 


import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
import nipype.interfaces.dipy as dipy
import nipype.interfaces.utility as utility
from nipype.interfaces.utility import IdentityInterface, Function
from os.path import join as opj
import os 
os.environ["PATH"] += os.pathsep + '/path/to/ants/bin'
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node, MapNode, JoinNode
import numpy as np
from IPython.display import Image
from nipype import config
cfg = dict(execution={'remove_unnecessary_outputs': False})
config.update_config(cfg)
#____________________#
#2
# Subjects list names, here I call 1R+ rat number; e.g. 1R01, 1R02, ...... 


Rat1 = ['1R'+str(i).zfill(2) for i in range(1,41)]

#____________________#
#3
#Defining directories, where to find data and where to generate outputs  

subjects = Rat1
experiment_dir = '/path/to/file/T1_work/'
output_dir  = 'DTI_output'
working_dir = 'DTI_workdir'

#____________________#
#4
# Defining a Nipype WorkFlow, that means how each step is connected with the next one, more information is on the website of nipype 
# Infosource - a function free node to iterate over the subjects list

DTI_workflow.base_dir = opj(experiment_dir, working_dir)

infosource = Node(IdentityInterface(fields=['subject_id']),
                  name="infosource")
infosource.iterables = [('subject_id', subjects)]

#match the data to your structured folder, the MOST important step to get it works
templates = {


 'DWI'  : 'bids/{subject_id}/dwi/{subject_id}_dwich.nii.gz',

 'Mask' : 'mask/{subject_id}_T2w_mask.nii.gz',

 'Mask_dti' : 'mask/{subject_id}_T2w_mask_.nii.gz',

 'topupB0':'bids/{subject_id}/dwi/{subject_id}__B0_topup.nii.gz',

 'T1w':'bids/{subject_id}/anat/{subject_id}_T1w.nii.gz',
 
 'T2w':'bids/{subject_id}/anat/{subject_id}_T2w.nii.gz'
 }

selectfiles = Node(SelectFiles(templates,
                               base_directory=experiment_dir, force_lists=["STM"]),
                   name="selectfiles")

#Pre-existing files, like bval, bvec,....etc change it accordingly
bval = '/path/to/file/dti.bval'
bvec = '/path/to/file/dti.bvec'
index = '/path/to/file/index_dti.csv'
acqparams = '/path/to/file/acqp.txt' 
Template = '/path/to/file/SELIM_Template_FA.nii'

#____________________#
#5
#Denoising Function 

def denoisefun(img):
  import numpy as np
  import os
  from dipy.io.image import load_nifti, save_nifti
  from dipy.denoise.patch2self import patch2self
  from dipy.denoise.localpca import mppca
  data, affine = load_nifti(img)
  denoised_arr = mppca(data,patch_radius=2)
  pathtosave=os.path.abspath(os.getcwd()+'/dwi_denoised.nii.gz')
  save_nifti(pathtosave, denoised_arr, affine)
  del data , affine, denoised_arr
  return pathtosave

#____________________#
#6
#Denoising Node


denoise = Node(name = 'denoise',
                  interface = Function(input_names = ['img'],
                  					   output_names = ['pathtosave'],
                  function = denoisefun))

#____________________#
#7
#Gibbs unringing function 

def gibssfun(img):
    from dipy.denoise.gibbs import gibbs_removal
    import numpy as np
    import os
    from dipy.io.image import load_nifti, save_nifti
    data, affine = load_nifti(img)
    denoised_arr = gibbs_removal(data)
    pathtosave=os.path.abspath(os.getcwd()+'/dwi_unringed.nii.gz')
    save_nifti(pathtosave, denoised_arr, affine)
    del data , affine, denoised_arr
    return pathtosave


#____________________#
#8
#Gibbs unringing Node 


gibssunringing= Node(name = 'gibssunringing',
                  interface = Function(input_names = ['img'],
                  					   output_names = ['pathtosave'],
                  function = gibssfun))
#____________________#
#9
# Topup function and node 
topup=Node(fsl.TOPUP(), name='topup')
#topup.inputs.in_file = "/path/to/file/{subject_id}__B0_topup.nii.gz"
topup.inputs.encoding_file = "/path/to/file/topup_encoding.txt"
topup.inputs.output_type = "NIFTI_GZ"
topup.inputs.config='/path/to/file/TOPUP_config.cnf'

#____________________#
#10
# APPLY TopUp function and node 

applytopup = Node(fsl.ApplyTOPUP(method="jac", in_index=[1]), name='applytopup')
applytopup.inputs.encoding_file = "/path/to/file/topup_encoding.txt"
applytopup.inputs.output_type = "NIFTI_GZ"

#____________________#
#11
# Eddy correction function and node 

eddy = Node (fsl.Eddy(), name = 'eddy')
eddy.inputs.in_acqp = "/path/to/file/acqp.txt"
eddy.inputs.in_bval = "/path/to/file/dwich.bval"
eddy.inputs.in_bvec = "/path/to/file/dwich.bvec"
eddy.inputs.in_index = "/path/to/file/index.csv"
eddy.inputs.output_type = "NIFTI_GZ"
#eddy.inputs.use_cuda = True
eddy.inputs.is_shelled = True
#eddy.inputs.outlier_nstd = 3
eddy.inputs.outlier_sqr = True
#eddy.inputs.outlier_type = 'both'
eddy.inputs.repol = True
eddy.inputs.num_threads = 8

#____________________#IF YOU DO NOT HAVE Re-PHASE COMMENT STEPS 9,10 & 11, USE 11(2) INSTEAD
#11(2)
# Eddy correction, old style (registration to b0)
# eddy = Node (fsl.EddyCorrect(), name = 'eddy')
# eddy.inputs.ref_num = 0


#____________________#
#12
#Get only b1000 volumes function
def get_dti(img):
   dtivol=[0,1,2]+[i+1 for i in range(2,42,2)]
   import os 
   import nibabel as nib
   img = nib.load(img)
   values = img.get_fdata().copy()[:,:,:,dtivol]
   header = img.header.copy()
   img_ = nib.Nifti1Image(values, img.affine, header)
   nib.save(img_, os.getcwd())
   img_dti= os.path.abspath(os.getcwd()+'.nii')
   del img,values, header, img_
   return img_dti


#____________________#
#13
#Get only b1000 volumes Node
dtivol = Node(name = 'dtivol',
                  interface = Function(input_names = ['img'],
                  					   output_names = ['img_dti'],
                  function = get_dti))

#____________________#
#14
#Fitting Tensor using RESTORE

dti_dipy=Node(dipy.RESTORE(), 'dti_dipy')
dti_dipy.inputs.in_bvec = '/path/to/file/DwGradVec_1000.txt'
dti_dipy.inputs.in_bval = '/path/to/file/DwEffBval_1000.txt'

#____________________#
#15
#Fitting Tensor using fsl DTIFIT


fit_tensor = Node (fsl.DTIFit(), name = 'fit_tensor')
fit_tensor.inputs.bvals = bval
fit_tensor.inputs.bvecs = bvec
fit_tensor.inputs.save_tensor = True
fit_tensor.inputs.args = '-w' #Fit the tensor with wighted least squares

#____________________#
#16
#Get the radial diffusivity by taking the first, the sum of L2 and L3
l2_l3_sum = Node(fsl.BinaryMaths(), name = 'l2_l3_sum')
l2_l3_sum.inputs.operation = 'add'
#Now the average
RD = Node (fsl.BinaryMaths(), name = 'RD')
RD.inputs.operand_value = 2
RD.inputs.operation = 'div'

#____________________#
#17
#Erosion before Registration
erod = Node (fsl.ErodeImage(), name = 'erod')
erod.inputs.kernel_shape= '2D'



#____________________#
#18
# Registration using ANTs

#Transform maps to Template
FA_to_Temp = Node(ants.Registration(), name = 'FA_to_Temp')
FA_to_Temp.inputs.args='--float'
FA_to_Temp.inputs.collapse_output_transforms=True
FA_to_Temp.inputs.initial_moving_transform_com=True
FA_to_Temp.inputs.fixed_image= Template
FA_to_Temp.inputs.num_threads=8
FA_to_Temp.inputs.output_inverse_warped_image=True
FA_to_Temp.inputs.output_warped_image=True
FA_to_Temp.inputs.sigma_units=['vox']*3
FA_to_Temp.inputs.transforms= ['Rigid', 'Affine', 'SyN']
FA_to_Temp.inputs.winsorize_lower_quantile=0.005
FA_to_Temp.inputs.winsorize_upper_quantile=0.995
FA_to_Temp.inputs.convergence_threshold=[1e-6]
FA_to_Temp.inputs.convergence_window_size=[10]
FA_to_Temp.inputs.metric=['MI', 'MI', 'CC']
FA_to_Temp.inputs.metric_weight=[1.0]*3
FA_to_Temp.inputs.number_of_iterations=[[1000, 500, 250, 100],
                                                 [1000, 500, 250, 100],
                                                 [100, 70, 50, 20]]
FA_to_Temp.inputs.radius_or_number_of_bins=[32, 32, 4]
FA_to_Temp.inputs.sampling_percentage=[0.25, 0.25, 1]
FA_to_Temp.inputs.sampling_strategy=['Regular',
                                              'Regular',
                                              'None']
FA_to_Temp.inputs.shrink_factors=[[8, 4, 2, 1]]*3
FA_to_Temp.inputs.smoothing_sigmas=[[3, 2, 1, 0]]*3
FA_to_Temp.inputs.transform_parameters=[(0.1,),
                                                 (0.1,),
                                                 (0.1, 3.0, 0.0)]
FA_to_Temp.inputs.use_histogram_matching=True
FA_to_Temp.inputs.write_composite_transform=True
FA_to_Temp.inputs.verbose=True
FA_to_Temp.inputs.output_warped_image=True
FA_to_Temp.inputs.float=True
#____________________#
#19
# Applying Registration

#>>>>MD
antsApplyMD_Study = Node(ants.ApplyTransforms(), name = 'antsApplyMD_Study')
antsApplyMD_Study.inputs.dimension = 3
antsApplyMD_Study.inputs.input_image_type = 3
antsApplyMD_Study.inputs.num_threads = 1
antsApplyMD_Study.inputs.float = True
antsApplyMD_Study.inputs.output_image = 'MD_{subject_id}.nii'
antsApplyMD_Study.inputs.reference_image = Study_Template


#>>>>AD
antsApplyAD_Study = antsApplyMD_Study.clone(name = 'antsApplyAD_Study')
antsApplyAD_Study.inputs.output_image = 'AD_{subject_id}.nii'


#>>>>RD
antsApplyRD_Study = antsApplyMD_Study.clone(name = 'antsApplyRD_Study')
antsApplyRD_Study.inputs.output_image = 'RD_{subject_id}.nii'





#____________________#
#20
# Connecting Workflow
DTI_workflow.connect ([

      (infosource, selectfiles,[('subject_id','subject_id')]),
      (selectfiles,denoise,[('Mask','mask')]),
      (selectfiles,denoise,[('DWI','img')]),    
      (denoise,gibssunringing,[('pathtosave','img')]),
#>>>>TopUp&Eddy
      (selectfiles, topup, [('topupB0','in_file')]),
      (topup, applytopup,[("out_fieldcoef", "in_topup_fieldcoef"),("out_movpar", "in_topup_movpar")]), 
      (gibssunringing,applytopup,[('pathtosave','in_files')]),
      (applytopup,eddy,[('out_corrected','in_file')]) ,
      (selectfiles, eddy, [('Mask','in_mask')]),
#>>>>Selecting B1000

      (eddy, dtivol, [('out_corrected','img')]),
 
#>>>>    Tensor fitting using DiPy RESTORE  
    #(dtivol, dti_dipy, [('img_dti','in_file')]),
    #(selectfiles, dti_dipy, [('Mask','in_mask')]),
#>>>>Tensor fitting Using FSL
     (selectfiles, fit_tensor, [('Mask_dti','mask')]),
     (dtivol, fit_tensor, [('img_dti','dwi')]),
     (fit_tensor, l2_l3_sum, [('L2','in_file')]),
     (fit_tensor, l2_l3_sum, [('L3','operand_file')]),
     (l2_l3_sum, RD, [('out_file','in_file')]),
     (fit_tensor, erod, [('FA','in_file')]),


#>>>>
# Reg using ANTS
      (erod, FA_to_Temp, [('out_file','moving_image')]),       

      (fit_tensor, antsApplyMD_WAX, [('MD','input_image')]),
      (FA_to_Temp, antsApplyMD_WAX, [('composite_transform','transforms')]),

      (fit_tensor, antsApplyAD_WAX, [('L1','input_image')]),
      (FA_to_Temp, antsApplyAD_WAX,[('composite_transform','transforms')]),

      (RD, antsApplyRD_WAX, [('out_file','input_image')]),
      (FA_to_Temp, antsApplyRD_WAX,[('composite_transform','transforms')]),      
 

   ])     
#____________________#
#21
#Run in parallel using all CPU cores
DTI_workflow.run('MultiProc',plugin_args={} )
