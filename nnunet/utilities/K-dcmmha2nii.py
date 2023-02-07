# Keenster 20230207 将文件夹下所有dcm和mha文件转为nii.gz，针对肌肉数据集MyoSegmenTUM_spine，以适合matlab计算
import os
import SimpleITK as sitk

path = "E:\Keenster Files\MedDatav2\MyoSegmenTUM_spine_KeensterRemap"
id=""
patient_name=""
dist_dir=""

for root, dirs, files in os.walk(path):
    print(root)
    print(dirs)
    print(files)
    if root.endswith("muscle"):
        id=root.split('\\')[-2]
        patient_name="Study_"+id
        dist_dir=path+"\\"+patient_name
        if not os.path.exists(dist_dir):
            os.makedirs(dist_dir)
        for file in files:
            if file.find("FATFRACTION")!=-1:
                reader = sitk.ImageFileReader()
                reader.SetFileName(root+"\\"+file)
                image = reader.Execute()
                writer = sitk.ImageFileWriter()
                writer.SetFileName(dist_dir+"\\"+patient_name+".nii.gz")
                writer.Execute(image)
            elif file.find(id)!=-1:
                reader = sitk.ImageFileReader()
                reader.SetFileName(root + "\\" + file)
                image = reader.Execute()
                writer = sitk.ImageFileWriter()
                writer.SetFileName(dist_dir+"\\"+patient_name +"_"+file[0:file.find(id)-1]+ ".nii.gz")
                writer.Execute(image)
    if root.endswith("vertebrae"):
        for file in files:
            if file.startswith("L"):
                reader = sitk.ImageFileReader()
                reader.SetFileName(root + "\\" + file)
                image = reader.Execute()
                writer = sitk.ImageFileWriter()
                writer.SetFileName(dist_dir + "\\" + file[0:file.find(".")] + ".nii.gz")
                writer.Execute(image)