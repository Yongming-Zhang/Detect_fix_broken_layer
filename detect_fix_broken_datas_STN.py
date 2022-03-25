import logging
import os
import sys
import numpy as np
import torch
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
import sys
import cv2
from unet_model import UNet
from unetwork import U_Network, SpatialTransformer
# sys.path.insert(0, '/mnt/users/code/main_code')
# from mitok.utils.mdicom import SERIES

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

#生成原始32层图像用来做配准和6层图像用来生成配准的fix图像
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

#生成配准的fix图像
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
    # if broken_index == 113:
    #     save_dcm_nii('/mnt/data/zhangyongming/lz/truth_broken_datas/case', fix_datas, '1287300_113_pred.nii.gz')

    return fix_datas

def MaxMinNormalization(x):
    Max = torch.max(x)
    Min = torch.min(x)
    x = (x - Min) / (Max - Min)
    
    return x, Max, Min

#生成moving image和fix image配准后的偏移量
def generate_registration_offsets(moving_datas, fix_datas, device):
    # offsets = np.zeros((1, 3, 32, fix_datas.shape[0], fix_datas.shape[1]))
    moving_datas = np.transpose(moving_datas, (2, 1, 0))
    vol_size = moving_datas.shape

    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 32, 16, 16]

    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    UNet.load_state_dict(torch.load('./checkpoints/15000.pth'))
    STN_img = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)
    UNet.eval()
    STN_img.eval()
    STN_label.eval()

    # 读入moving图像    
    input_moving = moving_datas[np.newaxis, np.newaxis, ...]
    input_moving = torch.from_numpy(input_moving).to(device).float()
    # 读入moving图像对应的label
    fix_datas = np.transpose(fix_datas, (2, 1, 0))
    input_label = fix_datas[np.newaxis, np.newaxis, ...]
    input_label = torch.from_numpy(input_label).to(device).float()

    input_moving, Max, Min = MaxMinNormalization(input_moving)
    input_fixed, Max, Min = MaxMinNormalization(input_label)
    # 获得配准后的图像和label
    pred_flow = UNet(input_moving, input_fixed)
    # print(pred_flow)
    print(pred_flow.shape, torch.max(pred_flow[:,2,30,:,:]), torch.min(pred_flow[:,2,30,:,:]))
    # pred_img = STN_img(input_moving, pred_flow)
    # pred_img = pred_img * (Max - Min) + Min
    # # print(pred_img.shape)
    # pred_label = STN_label(input_label, pred_flow)
    # # print(pred_label.shape)
    pred_flow = pred_flow.cpu().detach().numpy()
    # offsets[:, 0, :, :, :] = pred_flow[:, 2, :, :, :]
    # offsets[:, 1, :, :, :] = pred_flow[0, 1, :, :, :]
    # offsets[:, 2, :, :, :] = pred_flow[0, 0, :, :, :]

    return pred_flow, STN_img

def compute_offsets(broken_layer_index, all_offsets):
    all_xy_coordinates = []
    for k in range(len(broken_layer_index)):
        offsets = all_offsets[k]
        offsets = np.transpose(offsets, (0, 1, 4, 3, 2))
        offsets = offsets[::-1, ::-1, :]

        xy_offsets = np.zeros(offsets.shape)
        for i in range(offsets.shape[0]):
            for j in range(offsets.shape[1]):
                # print(np.array([i, j]))
                # print('offsets', offsets[i, j])
                xy_offsets[i, j] = (i, j) - offsets[i, j]
                # print('xy_offsets', xy_offsets[i, j])
        all_xy_coordinates.append(xy_offsets)
    
    return all_xy_coordinates

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
    broken_offsets = np.zeros((1, 3, 32, 512, 512))
    for broken_index in broken_layer_index:
        moving_datas, six_datas = divide_datas(broken_index, dcm_path, all_dcm_length, min_dcm_start_value)
        fix_datas = generate_fix_datas(broken_index, moving_datas, six_datas, device)
        # if broken_index == 113:
        #     save_dcm_nii('/mnt/data/zhangyongming/lz/truth_broken_datas/case', fix_datas, broken_layer_name.split('/')[0]+'_'+str(broken_index)+'fix.nii.gz')
        offsets, STN_img = generate_registration_offsets(moving_datas, fix_datas, device)
        broken_offsets += offsets
        all_offsets.append(broken_offsets)

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
            # if l+1 - broken_layer_index[start_index] < 32 and l+1 != broken_layer_index[start_index]:
            #     moving_datas[:, :, 31-(l+1-broken_layer_index[start_index])] = all_moved_datas[start_index]   
            moving_datas = np.transpose(moving_datas, (2, 0, 1))[np.newaxis, np.newaxis, ...]
            moving_datas = torch.from_numpy(moving_datas).to(device).float()
            # print('moving_datas', torch.max(moving_datas), torch.min(moving_datas))
            # moving_datas, Max, Min = MaxMinNormalization(moving_datas)
            # print('Max, Min', Max, Min)
            # if l+1 == 113:
            #     save_dcm_nii('/mnt/data/zhangyongming/lz/truth_broken_datas/case', np.transpose(moving_datas[0, 0, :, :, :].cpu().detach().numpy(), (2, 1, 0)), '1287300_113_prestn.nii.gz')
            pred_flow = torch.from_numpy(all_offsets[start_index]).to(device).float()
            moved_datas = STN_img(moving_datas, pred_flow)
            # moved_datas = moved_datas * (Max - Min) + Min
            # print('STN_img', torch.max(moving_datas), torch.min(moving_datas))
            data = moved_datas[0, 0, 31, :, :].cpu().detach().numpy()
            # if l+1 == 113:
            #     save_dcm_nii('/mnt/data/zhangyongming/lz/truth_broken_datas/case', np.transpose(moved_datas[0, 0, :, :, :].cpu().detach().numpy(), (2, 1, 0)), broken_layer_name.split('/')[0]+'_'+str(l+1)+'moved.nii.gz')
            # data = np.transpose(data, (1, 0))
            # print('l+1>=', broken_layer_index[start_index], l)
            # if l+1 == broken_layer_index[start_index]:
            #     all_moved_datas.append(data)
            origin_datas[:, :, l] = data
        if start_index+1 < len(broken_layer_index):
            if l+1 == broken_layer_index[start_index+1]-1:
                start_index += 1
                # print('start_index', start_index)

        pid_sid = dcm_path.split('/')[6] + '_' + dcm_path.split('/')[7] + '_' + dcm_path.split('/')[8] 
        new_dcm_path = os.path.join('/mnt/data/zhangyongming/lz/truth_broken_datas/fix_truth_broken_datas', pid_sid) 
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
    # broken_layer_data_dir = [broken_layer_data_dir[0]]

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
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