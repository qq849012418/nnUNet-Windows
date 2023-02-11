# Keenster 230211 思路：先将所有的label图融合为一个图，然后将新label和原影像改名后保存到特定文件夹
import numpy as np
from collections import OrderedDict

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import SimpleITK as sitk
import shutil
import os

def combine_labels(f1,f2,f3,f4, out_file):

    img1 = sitk.ReadImage(f1)
    img_npy1 = sitk.GetArrayFromImage(img1)
    img2 = sitk.ReadImage(f2)
    img_npy2 = sitk.GetArrayFromImage(img2)
    img3 = sitk.ReadImage(f3)
    img_npy3 = sitk.GetArrayFromImage(img3)
    img4 = sitk.ReadImage(f4)
    img_npy4 = sitk.GetArrayFromImage(img4)

    uniques = np.unique(img_npy1)
    for u in uniques:
        if u not in [0, 1]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy1)
    seg_new[img_npy1 == 1] = 1
    seg_new[img_npy2 == 1] = 2
    seg_new[img_npy3 == 1] = 3
    seg_new[img_npy4 == 1] = 4
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img1)
    sitk.WriteImage(img_corr, out_file)


if __name__ == "__main__":

    path = "E:\Keenster Files\MedDatav2\MyoSegmenTUM_spine_KeensterRemap"

    task_name = "Task502_Study45-9"

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesVal = join(target_base, "imagesVal")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesVal)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    patient_names = []
    trainset=[]
    testset=[]

    for root in os.walk(path):
        # print(root)
        # print(dirs)
        # print(files)
        if not root.split('\\')[-1].startswith("Study_"):
            continue
        ori_img=join(root,root.split('\\')[-1]+".nii.gz")
        seg_img=

        id=root[-2]
        if int(id)%6==0:
            testset.append(id)
        else:
            trainset.append(id)
            shutil.copy(t1, join(target_imagesVal, patient_name + "_0000.nii.gz"))



    json_dict = OrderedDict()
    json_dict['name'] = "k"
    json_dict['description'] = "k"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "k"
    json_dict['licence'] = "k"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "FatFraction"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "psoas_left",
        "2": "psoas_right",
        "3": "erector_spinae_left",
        "4": "erector_spinae_right"
    }
    json_dict['numTraining'] = len(trainset)
    json_dict['numTest'] = len(testset)
    json_dict['training'] = [{'image': "./imagesTr/Study_%s.nii.gz" % i, "label": "./labelsTr/Study_%s.nii.gz" % i} for i in
                             trainset]
    json_dict['test'] = ["./imagesTs/Study_%s.nii.gz" % i for i in testset]

    save_json(json_dict, join(target_base, "dataset.json"))

    for p in subdirs(downloaded_data_dir, join=False):
        patdir = join(downloaded_data_dir, p)
        patient_name = p
        t1 = join(patdir, p + "_t1.nii.gz")
        t1c = join(patdir, p + "_t1ce.nii.gz")
        t2 = join(patdir, p + "_t2.nii.gz")
        flair = join(patdir, p + "_flair.nii.gz")

        assert all([
            isfile(t1),
            isfile(t1c),
            isfile(t2),
            isfile(flair),
        ]), "%s" % patient_name

        shutil.copy(t1, join(target_imagesVal, patient_name + "_0000.nii.gz"))
        shutil.copy(t1c, join(target_imagesVal, patient_name + "_0001.nii.gz"))
        shutil.copy(t2, join(target_imagesVal, patient_name + "_0002.nii.gz"))
        shutil.copy(flair, join(target_imagesVal, patient_name + "_0003.nii.gz"))

    """
    #I dont have the testing data
    downloaded_data_dir = "/home/fabian/Downloads/BraTS2018_train_val_test_data/MICCAI_BraTS_2018_Data_Testing_FIsensee"

    for p in subdirs(downloaded_data_dir, join=False):
        patdir = join(downloaded_data_dir, p)
        patient_name = p
        t1 = join(patdir, p + "_t1.nii.gz")
        t1c = join(patdir, p + "_t1ce.nii.gz")
        t2 = join(patdir, p + "_t2.nii.gz")
        flair = join(patdir, p + "_flair.nii.gz")

        assert all([
            isfile(t1),
            isfile(t1c),
            isfile(t2),
            isfile(flair),
        ]), "%s" % patient_name

        shutil.copy(t1, join(target_imagesTs, patient_name + "_0000.nii.gz"))
        shutil.copy(t1c, join(target_imagesTs, patient_name + "_0001.nii.gz"))
        shutil.copy(t2, join(target_imagesTs, patient_name + "_0002.nii.gz"))
        shutil.copy(flair, join(target_imagesTs, patient_name + "_0003.nii.gz"))"""