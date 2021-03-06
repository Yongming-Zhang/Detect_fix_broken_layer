import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import DataLoader

import monai
from monai.data import CSVSaver, ImageDataset
from monai.transforms import AddChannel, Compose, Resize, ScaleIntensity, EnsureType
from monai.transforms.croppad.array import CenterSpatialCrop, RandSpatialCrop
from resnet_3d import resnet50
import nibabel as nib
import pydicom
import torch.nn.functional as F
import time
import cv2
from unet_model import UNet

os.environ['VXM_BACKEND'] = 'pytorch'
sys.path.insert(0, '/mnt/users/code/voxelmorph')
from voxelmorph.torch.networks import VxmDense
from voxelmorph.torch.losses import NCC, MSE, Grad
from voxelmorph.py.utils import *
from voxelmorph.generators import *

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow, device):
        # new locations
        new_locs = self.grid.to(device) + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

def save_dcm_nii(save_path, data, name):
    # data = data[::-1,::-1,...]
    # data = np.transpose(data, (1, 0, 2))
    # print(np.unique(data))
    affine_arr = np.eye(4)
    plaque_nii = nib.Nifti1Image(data, affine_arr)
    nib.save(plaque_nii, os.path.join(save_path, name)) 

def save_dcm(save_path, data, template_dcm_file):
    ds = pydicom.dcmread(template_dcm_file)
    datai = data.astype(np.int16)
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        ds.decompress()
    ds.PixelData = datai.tostring()
    ds.fix_meta_info()
    ds.save_as(save_path, write_like_original=False) 

def load_dicom(l, dcm_path, min_dcm_start_value):
    
    if l+min_dcm_start_value < 10:
        dcm_path = os.path.join(dcm_path, str(l+min_dcm_start_value)+'.dcm')
    elif l+min_dcm_start_value >= 10 and l+min_dcm_start_value < 100:
        dcm_path = os.path.join(dcm_path, str(l+min_dcm_start_value)+'.dcm')
    elif l+min_dcm_start_value >= 100 and l+min_dcm_start_value < 1000:
        dcm_path = os.path.join(dcm_path, str(l+min_dcm_start_value)+'.dcm')
    # dcm_path = os.path.join(dcm_path, str(l+1)+'.dcm')
    #print(dcm_path)
    ds = pydicom.dcmread(dcm_path)
    data = ds.pixel_array

    return data, dcm_path 

def get_truth_data(dcm_path):
    # dcm_length = len(os.listdir(dcm_path))
    vessel_mask_path = os.path.join(dcm_path+'_CTA', 'mask_source/mask_vessel.nii.gz')
    vessel_mask_data = nib.load(vessel_mask_path)
    vessel_mask_data = vessel_mask_data.get_fdata()

    vessel_mask_data_points = np.argwhere(vessel_mask_data==1)
    xyz_max = np.max(vessel_mask_data_points, axis=0) 
    xyz_min = np.min(vessel_mask_data_points, axis=0) 
    print('xyz_max, xyz_min', xyz_max, xyz_min)
    x_max, y_max, z_max = xyz_max
    x_min, y_min, z_min = xyz_min
    if z_min < 31:
        z_min = 31

    all_dcm_length = len(os.listdir(dcm_path))
    z_max = all_dcm_length - 17

    dcm_list = os.listdir(dcm_path)
    dcm_list.sort()
    min_dcm_start_value = int(dcm_list[0].split('.')[0])
    print('min_dcm_start_value', min_dcm_start_value)
    
    datas = np.zeros((512, 512, all_dcm_length))
    for l in range(all_dcm_length):
        data, _ = load_dicom(l, dcm_path, min_dcm_start_value)    
        datas[:,:,l] = data

    return datas, z_min, z_max

def detect_broken_index(broken_layer_data_path, broken_layer_name, device):
    test_data, z_min, z_max = get_truth_data(os.path.join(broken_layer_data_path, broken_layer_name))
    # print(test_data.shape)
    
    scaleIntensity = ScaleIntensity()
    ensureType = EnsureType()
    centerSpatialCrop = CenterSpatialCrop(roi_size=(256, 256))
    randSpatialCrop = RandSpatialCrop(random_size=False, roi_size=(224, 224))
    resize = Resize((256, 256))

    img_data = np.transpose(test_data, (2, 0, 1))
    print('img_data', img_data.shape)        

    model = resnet50(spatial_dims=3, n_input_channels=1, num_classes=2, out_channels=6).to(device)

    model.load_state_dict(torch.load("./checkpoints/model_axis_datasets_resnet50_epoch20_20220104.pth"))
    model.eval()
    probability = 0
    broken_index = []
    with torch.no_grad():
        for i in range(z_min, z_max-7):
            image = img_data[i:i+6, :, :]
            # image = image[:, ::-1, :]
            image = np.transpose(image, (0, 2, 1))
            image = centerSpatialCrop(image)
            image = scaleIntensity(image)
            image = ensureType(image)      
                
            image = torch.FloatTensor(image)
            image = torch.unsqueeze(image, dim=0)
            image = torch.unsqueeze(image, dim=0)
            image = image.to(device)

            outputs = model(image)
            # print('outputs', outputs.size(), outputs)
            # pred = F.softmax(outputs, dim=1)

            outputs = F.softmax(outputs, dim=1)
            # print(outputs)
            pred = outputs
            pred_outputs = outputs.argmax(dim=1)
            # print(pred_outputs)

            if torch.any(pred_outputs != 0):
                ind = pred.argmax(dim=1).item()
                if ind == 0:
                    continue
                # if pred[0, ind] > 0.9:
                index = ind + i + 1
                print(i, index)
                if pred[0, ind] > probability:
                    probability = pred[0, ind]
                if index not in broken_index:
                    broken_index.append(index)

    return probability, sorted(broken_index)

#????????????32???????????????????????????6??????????????????????????????fix??????
def divide_datas(broken_index, dcm_path, all_dcm_length, min_dcm_start_value):
    broken_index = broken_index - 1 
    start_broken_index = broken_index - 31
    for l in range(all_dcm_length):
        if l == 0:
            data, origin_dcm_path = load_dicom(l, dcm_path, min_dcm_start_value)
            moving_datas = np.zeros((data.shape[0], data.shape[1], 32))
        if l >= start_broken_index and l <= int(broken_index):
            data, origin_dcm_path = load_dicom(l, dcm_path, min_dcm_start_value)
            moving_datas[:, :, l-start_broken_index] = data
    moving_datas = np.transpose(moving_datas, (1, 0, 2))
    six_datas = moving_datas[:, :, 25:31]
    
    return moving_datas, six_datas

#???????????????fix??????
def generate_fix_datas(broken_index, moving_datas, six_datas, device):
    net = UNet(n_channels=6, n_classes=1, bilinear=True)
    net.load_state_dict(torch.load('/mnt/users/code/unet/checkpoints/unet_0314/model_epoch300.pth', map_location=device))
    net.to(device)

    imgs = np.transpose(six_datas, (2, 0, 1))
    imgs = torch.FloatTensor(imgs)
    # print(imgs.shape)
    imgs = torch.unsqueeze(imgs, dim=0).to(device)
    imgs, Max, Min = MaxMinNormalization(imgs)
    mask_pred = net(imgs)
    mask_pred = mask_pred * (Max - Min) + Min
    # print(mask_pred.shape)
    mask_pred = torch.squeeze(mask_pred, dim=0).cpu().detach().numpy()
    mask_pred = np.transpose(mask_pred, (1, 2, 0))

    fix_datas = np.zeros(moving_datas.shape)
    fix_datas[:, :, :31] = moving_datas[:, :, :31]
    fix_datas[:, :, 31:32] = mask_pred
    # fix_datas[:, :, 31:32] = moving_datas[:, :, 30:31]

    return fix_datas

def MaxMinNormalization(x):
    Max = torch.max(x)
    Min = torch.min(x)
    x = (x - Min) / (Max - Min)
    
    return x, Max, Min

def MaxMinNormalizations(x):
    Max = np.max(x)
    Min = np.min(x)
    x = (x - Min) / (Max - Min)
    
    return x, Max, Min

def load_volfiles(
    data,
    np_var='vol',
    add_batch_axis=False,
    add_feat_axis=False,
    pad_shape=None,
    resize_factor=1,
    ret_affine=False
):
    """
    Loads a file in nii, nii.gz, mgz, npz, or npy format. If input file is not a string,
    returns it directly (allows files preloaded in memory to be passed to a generator)

    Parameters:
        filename: Filename to load, or preloaded volume to be returned.
        np_var: If the file is a npz (compressed numpy) with multiple variables,
            the desired variable can be specified with np_var. Default is 'vol'.
        add_batch_axis: Adds an axis to the beginning of the array. Default is False.
        add_feat_axis: Adds an axis to the end of the array. Default is False.
        pad_shape: Zero-pad the array to a target shape. Default is None.
        resize: Volume resize factor. Default is 1
        ret_affine: Additionally returns the affine transform (or None if it doesn't exist).
    """
    
    vol = data.squeeze()
    affine = np.eye(4)

    if pad_shape:
        vol, _ = pad(vol, pad_shape)

    if add_feat_axis:
        vol = vol[..., np.newaxis]

    if resize_factor != 1:
        vol = resize(vol, resize_factor)

    if add_batch_axis:
        vol = vol[np.newaxis, ...]

    return (vol, affine) if ret_affine else vol

def fix_register(movings, fixed, device):
    # load moving and fixed images
    moving = load_volfiles(movings, add_batch_axis=True, add_feat_axis=True)
    fixed, fixed_affine = load_volfiles(
        movings, add_batch_axis=True, add_feat_axis=True, ret_affine=True)

    model = '/mnt/users/code/voxelmorph/scripts/torch/models/model_0317/0100.pth'
    # device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    # load and set up model
    model = VxmDense.load(model, device)
    model.to(device)
    model.eval()

    moving, Max, Min = MaxMinNormalizations(moving)
    fixed, _, _ = MaxMinNormalizations(fixed)

    # set up tensors and permute
    input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
    input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)

    # predict
    moved, warp = model(input_moving, input_fixed, registration=True)

    # save moved image
    moved = moved.detach().cpu().numpy().squeeze()
    moved = moved * (Max - Min) + Min
    # save_volfile(moved, name, fixed_affine)
    # broken_layer_name = '1287300/9C7195D2/3443F930'
    # new_dcm_path = '/mnt/data/zhangyongming/lz/truth_broken_datas/fix_broken_nii/' + broken_layer_name.split('/')[0]
    # save_dcm_nii(new_dcm_path, moved, broken_layer_name.split('/')[0]+'_'+str(113)+'.nii.gz')
    
    return warp

def fix_broken_layer(broken_layer_data_path, broken_layer_name, origin_broken_layer_index, device):
    dcm_path = os.path.join(broken_layer_data_path, broken_layer_name)
    all_dcm_length = len(os.listdir(dcm_path))
    dcm_list = os.listdir(dcm_path)
    dcm_list.sort()
    min_dcm_start_value = int(dcm_list[0].split('.')[0])
    broken_layer_index = [origin_broken_layer_index[0]]
    for index in origin_broken_layer_index[1:]:
        index = int(index)
        if index - broken_layer_index[-1] > 31:
            broken_layer_index.append(index)
    all_offsets = []
    broken_offsets = torch.zeros((1, 3, 512, 512, 32))
    for broken_index in broken_layer_index:
        moving_datas, six_datas = divide_datas(broken_index, dcm_path, all_dcm_length, min_dcm_start_value)
        fix_datas = generate_fix_datas(broken_index, moving_datas, six_datas, device)       
        offsets = fix_register(moving_datas, fix_datas, device).detach().cpu().numpy()
        # broken_offsets += offsets
        all_offsets.append(offsets)

    print('broken_layer_index', broken_layer_index)
    # all_xy_coordinates = compute_offsets(broken_layer_index, all_offsets)
    start_index = 0
    origin_datas = np.zeros((512, 512, all_dcm_length))
    for l in range(all_dcm_length):
        data, origin_dcm_path = load_dicom(l, dcm_path, min_dcm_start_value)   
        origin_datas[:, :, l] = data
    # all_moved_datas = []
    for l in range(all_dcm_length):
        if l+1 < broken_layer_index[0]:
            data, origin_dcm_path = load_dicom(l, dcm_path, min_dcm_start_value)   
            # print('l+1<', broken_layer_index[0], l)         
        elif l+1 >= broken_layer_index[start_index]:
            data, origin_dcm_path = load_dicom(l, dcm_path, min_dcm_start_value)
            # data = cv2.remap(np.array(data, np.float32), np.array(all_xy_coordinates[start_index][:, :, 1], np.float32), np.array(all_xy_coordinates[start_index][:, :, 0], np.float32), cv2.INTER_LINEAR)
            moving_datas = origin_datas[:, :, l-31:l+1]  
            inshape = moving_datas.shape
            
            moving_datas = load_volfiles(moving_datas, add_batch_axis=True, add_feat_axis=True)
            moving_datas = torch.from_numpy(moving_datas).to(device).float().permute(0, 4, 1, 2, 3)
            moving_datas, Max, Min = MaxMinNormalization(moving_datas)

            pred_flow = torch.from_numpy(all_offsets[start_index]).to(device)
            spatialTransformer = SpatialTransformer(inshape)
            moved_datas = spatialTransformer(moving_datas, pred_flow, device)

            moved_datas = moved_datas * (Max - Min) + Min
            # print('STN_img', torch.max(moving_datas), torch.min(moving_datas))
            data = moved_datas.detach().cpu().numpy().squeeze()[:, :, 31]            
            origin_datas[:, :, l] = data
        if start_index+1 < len(broken_layer_index):
            if l+1 == broken_layer_index[start_index+1]-1:
                start_index += 1
                # print('start_index', start_index)

        pid_sid = dcm_path.split('/')[6] + '_' + dcm_path.split('/')[7] + '_' + dcm_path.split('/')[8] 
        new_dcm_path = os.path.join('/mnt/data/zhangyongming/lz/truth_broken_datas/fix_truth_broken_voxelmorph', pid_sid) 
        if not os.path.exists(new_dcm_path):
            os.makedirs(new_dcm_path)
        save_dcm(os.path.join(new_dcm_path, str(l+min_dcm_start_value)+'.dcm'), data, origin_dcm_path)
    
def main():
    # monai.config.print_config()
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    broken_layer_data_path = '/mnt/DrwiseDataNFS/drwise_runtime_env0416fix1_1222/data1/inputdata' 
    broken_layer_data_dir = ['1287300/9C7195D2/3443F930', '1287303/72FE4FE1/062EE826', '1287302/F9E9D907/8CB01A64',
    'AW1183694326.357.1623962658/1D20FDF3/3ABEBFDC', '1334004/B937AB30/B937AB32', '1287305/6C2BF372/10AA4CEA', 
    '1287301/71A19987/798AC981', 'lz01780/1BA48F4C/C4BDF72A', 'lz01708/89CC1851/472755D7', 'lz01525/766488C0/6F656322', 'lz01091/3055C960/7208FF02', 'lz00751/36F55DF3/4F15BB32', 
    'lz01674/7679B8CC/BEEEC38D', 'lz00422/4DAEAFF7/68274FBC', 'lz01536/5BFEE925/57E94268', 'lz01146/FB44FB28/3DF2DECA', 'lz01084/D35715F3/59E972B6', 'lz00325/3EFE733F/ACA60F72', 'lz01838/964AE090/2F150A32', 
    'lz01428/F972548C/662F6F14', 'lz00262/9E9C27ED/9E0AFB62', 'lz01692/1192DF74/F858DFFC', 'lz01056/347ADABC/012F295E', 'lz00464/6F8E820E/C5F6416C', 'lz00421/F2BCB29E/365A913C', 'lz00338/AB6D8322/86195141', 
    'lz00765/5ECFED77/EA8A793F', 'lz00868/6D8E837F/43FBAD66', 'lz00382/98089BE5/51D66757', 'lz01046/2FEC1B1B/E551CE98', 'lz00619/6E7133CE/96F385D1', 'lz01126/36874D57/B1E9619B', 'lz01160/14F257BE/A1BB6E1E', 
    'lz00256/1957E16D/22E5482C', 'lz00715/5DD2A85E/33CA13B7', 'lz01081/BC910B89/DC7C5B67', 'lz00614/B0B17CFB/4A463519', 'lz00376/FA0C0C29/3FFBE129', 'lz00157/DCF607E3/7793A8E9', 'lz00327/66D35F8A/EFBBC5D0'] #'lz00327/66D35F8A/EFBBC5D0'
    # broken_layer_data_dir = ['1287300/9C7195D2/3443F930', '1287303/72FE4FE1/062EE826', '1287302/F9E9D907/8CB01A64',
    #     'AW1183694326.357.1623962658/1D20FDF3/3ABEBFDC', '1334004/B937AB30/B937AB32', '1287305/6C2BF372/10AA4CEA', 
    #     '1287301/71A19987/798AC981'] 
    broken_layer_data_dir = [broken_layer_data_dir[0]]

    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    all_detect_broken_layer_time = []
    all_fix_broken_layer_time = []
    no_broken_layer_list = []
    for broken_layer_name in broken_layer_data_dir:
        start = time.time()
        print(os.path.join(broken_layer_data_path, broken_layer_name))
        
        probability, broken_index = detect_broken_index(broken_layer_data_path, broken_layer_name, device)
        print('max probability, broken_index', probability, broken_index)

        detect_broken_layer_time = time.time() - start
        print('detect broken layer time', detect_broken_layer_time)
        all_detect_broken_layer_time.append(detect_broken_layer_time)

        if broken_index != []:
            fix_broken_layer(broken_layer_data_path, broken_layer_name, broken_index, device)
        print('fix broken layer time', time.time()-start)
        all_fix_broken_layer_time.append(time.time()-start)

        if broken_index == []:
            no_broken_layer_list.append(broken_layer_name)
            print('There is no broken layer!')
            continue

    print('mean detect broken layer time', np.mean(all_detect_broken_layer_time))
    print('mean fix broken layer tmie', np.mean(all_fix_broken_layer_time))
    print('no broken layer list', no_broken_layer_list)

if __name__ == "__main__":
    main()