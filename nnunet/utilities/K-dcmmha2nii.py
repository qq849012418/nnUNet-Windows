# Keenster 20230207 将文件夹下所有dcm和mha文件转为nii.gz，针对肌肉数据集MyoSegmenTUM_spine，以适合matlab计算
import os
import SimpleITK as sitk

# 引用方法原文链接：https://www.jb51.cc/python/1119618.html
def resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkNearestNeighbor):
    """
    用itk方法将原始图像resample到与目标图像一致
    :param ori_img: 原始需要对齐的itk图像
    :param target_img: 要对齐的目标itk图像
    :param resamplemethod: itk插值方法: sitk.sitkLinear-线性  sitk.sitkNearestNeighbor-最近邻
    :return:img_res_itk: 重采样好的itk图像
    """
    target_Size = target_img.GetSize()      # 目标图像大小  [x,y,z]
    target_Spacing = target_img.GetSpacing()   # 目标的体素块尺寸    [x,y,z]
    target_origin = target_img.GetOrigin()      # 目标的起点 [x,y,z]
    target_direction = target_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]
    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(target_Size)		# 目标图像大小
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    # 根据需要重采样图像的情况设置不同的dype
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt16)   # 近邻插值用于mask的，保存uint16
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
    return itk_img_resampled


# 以下为主程序
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
            elif file.endswith(".mha"):
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
                image_r=resize_image_itk(image,sitk.ReadImage(dist_dir+"\\"+patient_name+".nii.gz"))#因为脊柱和原图的方向是不一样的，建议转！虽然这样会损失掉部分
                writer = sitk.ImageFileWriter()
                writer.SetFileName(dist_dir + "\\" + file[0:file.find(".")] + ".nii.gz")
                writer.Execute(image_r)
                # writer.Execute(image)



