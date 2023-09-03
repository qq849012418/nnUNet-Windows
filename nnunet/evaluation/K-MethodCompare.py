# ref: https://blog.csdn.net/weixin_44025103/article/details/131959640?ops_request_misc=&request_id=&biz_id=102&utm_term=prediction%20mask&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-131959640.142^v93^insert_down28v1&spm=1018.2226.3001.4187

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import torch


def get_tp(mask, prd):
    # tp 取共同的像素，矩阵相乘
    tp = mask * prd
    return tp

def get_fp(mask, prd):
    # # fp prd中去除mask的部分
    fp = ((1 - mask) * prd)
    return fp

def get_fn(mask, prd):
    # FN 取mask去掉prd的部分
    fn = (mask * (1 - prd))
    return fn
# def get_background(image, tp, fp, fn):
#     tp_fp_fn = tp + fp + fn
#     background = image * (1 - tp_fp_fn)
#     return background
#     pass


def overlay_images(mask, prd, tp_color, fp_color, fn_color, tp_alpha, fp_alpha, fn_alpha):
    # Create an RGB image with custom overlay colors and transparency
    overlay = np.zeros(mask.shape + (4,), dtype=np.uint8)

    # Set TP pixels
    overlay[prd > 0] = tp_color + [int(tp_alpha * 255)]

    # Set FP pixels
    overlay[(prd == 0) & (mask == 1)] = fp_color + [int(fp_alpha * 255)]

    # Set FN pixels
    overlay[(prd == 1) & (mask == 0)] = fn_color + [int(fn_alpha * 255)]

    return overlay

def main():
    mask_path = r'E:\KeensterFiles\MedDatav2\forCmopare\Study45-9_gt\Study_06_erector_spinae_left.nii.gz'  # Replace with the path to your mask image
    prd_path = r'E:\KeensterFiles\MedDatav2\forCmopare\Study_gtsam\Study_06_erector_spinae_left.nii.gz'  # Replace with the path to your prediction image
    background_path = r'D:\Keenster\MatlabScripts\KeensterSSM\Study_06\Study_06.nii.gz'  # Replace with the path to your background image

    selected_slice = 20  # Specify the desired z-slice
    tp_color = [255, 255, 0]  # Green color for TP
    fp_color = [255, 0, 0]  # Red color for FP
    fn_color = [0, 255,0]  # Blue color for FN
    tp_alpha = 0.5  # Transparency for TP
    fp_alpha = 0.5  # Transparency for FP
    fn_alpha = 0.5  # Transparency for FN
    background_darken_factor = 0.25  # Factor to darken the background image
    cropping_ratio = 0.5  # Cropping ratio (0.8 means 80% of the original size)


    mask = sitk.ReadImage(mask_path)
    prd = sitk.ReadImage(prd_path)
    background = sitk.ReadImage(background_path)

    # Select the desired z-slice for both mask and prediction
    mask = sitk.GetArrayFromImage(mask[:, :, selected_slice])
    prd = sitk.GetArrayFromImage(prd[:, :, selected_slice])
    background = sitk.GetArrayFromImage(background[:, :, selected_slice])
    # Normalize background pixel values to [0, 255]
    background = ((background - background.min()) / (background.max() - background.min()) * 255).astype(np.uint8)

    # Calculate cropping dimensions
    crop_height, crop_width = mask.shape[0], mask.shape[1]
    center_h, center_w = crop_height // 2, crop_width // 2
    new_height = int(crop_height * cropping_ratio)
    new_width = int(crop_width * cropping_ratio)
    top_h = center_h - new_height // 2
    left_w = center_w - new_width // 2
    bottom_h = top_h + new_height
    right_w = left_w + new_width

    # Crop mask, prediction, and background to the specified ratio
    mask = mask[top_h:bottom_h, left_w:right_w]
    prd = prd[top_h:bottom_h, left_w:right_w]
    background = background[top_h:bottom_h, left_w:right_w]

    tp = get_tp(mask, prd)
    fp = get_fp(mask, prd)
    fn = get_fn(mask, prd)

    overlay = overlay_images(mask, prd, tp_color, fp_color, fn_color, tp_alpha, fp_alpha, fn_alpha)

    # Display the images using matplotlib
    plt.figure(figsize=(12, 4))
    plt.subplot(141)
    plt.imshow(background, cmap='gray')
    plt.title('Image')

    plt.subplot(142)
    plt.imshow(mask, cmap='gray')
    plt.title('Ground Truth')

    plt.subplot(143)
    plt.imshow(prd, cmap='gray')
    plt.title('Prediction')


    plt.subplot(144)
    plt.imshow(background, cmap='gray', vmin=255*background_darken_factor,vmax=255)
    plt.imshow(overlay, alpha=1)
    plt.title('Custom Overlay (TP, FP, FN) on Background')

    plt.tight_layout()
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()



