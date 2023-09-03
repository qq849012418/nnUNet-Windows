# ref: https://blog.csdn.net/weixin_44025103/article/details/131959640?ops_request_misc=&request_id=&biz_id=102&utm_term=prediction%20mask&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-131959640.142^v93^insert_down28v1&spm=1018.2226.3001.4187

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os


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
# 用于单独保存子图的函数 https://blog.csdn.net/qq_39645262/article/details/127190982
def save_subfig(fig,ax,save_path,fig_name):
    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).expanded(1.02, 1.02)
    extent = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(save_path,fig_name), bbox_inches=extent)

def main():
    mask_base_path = r'E:\KeensterFiles\MedDatav2\forCmopare'
    gt_name = 'Study45-9_gt'
    algo_name_list = ['Study45-9_DeepSSM','Study_gtacm','Study_gtsam','Study45-9_ssm','Study45-9_ssmrcf']
    patient_id = 'Study_12'

    muscle_name_list = ["psoas_left", "psoas_right", "erector_spinae_left", "erector_spinae_right"]
    # muscle_name_list=["PML","PMR","ESL","ESR","MFL","MFR"]

    selected_slice = 20  # Specify the desired z-slice
    tp_color = [255, 255, 0]  # Yellow color for TP
    fp_color = [255, 0, 0]  # Red color for FP
    fn_color = [0, 255,0]  # Green color for FN
    tp_alpha = 0.5  # Transparency for TP
    fp_alpha = 0.5  # Transparency for FP
    fn_alpha = 0.5  # Transparency for FN
    background_darken_factor = 0.25  # Factor to darken the background image https://blog.csdn.net/xvvoly/article/details/105157455
    cropping_ratio = 0.5  # Cropping ratio (0.8 means 80% of the original size)
    img_out_path=os.path.join(mask_base_path,'ResultVisualization')
    if not os.path.exists(img_out_path):
        os.makedirs(img_out_path)

    background_path = 'D:\\Keenster\\MatlabScripts\\KeensterSSM\\'+patient_id+'\\'+patient_id+'.nii.gz'  # Replace with the path to your background image
    background = sitk.ReadImage(background_path)
    background = sitk.GetArrayFromImage(background[:, :, selected_slice])
    # Normalize background pixel values to [0, 255]
    background = ((background - background.min()) / (background.max() - background.min()) * 255).astype(np.uint8)

    # Display the images using matplotlib
    fig=plt.figure(figsize=(12,4))
    # plt.subplot(121)
    # plt.imshow(background_crop, cmap='gray')
    # # plt.title('Image')
    algosize=len(algo_name_list)
    for i in range(algosize):
        algo_name = algo_name_list[i]
        overlay_list=[]
        for mus_name in muscle_name_list:
            mask_path = os.path.join(mask_base_path,gt_name,patient_id+"_"+mus_name+".nii.gz") # Replace with the path to your mask image
            prd_path = os.path.join(mask_base_path,algo_name,patient_id+"_"+mus_name+".nii.gz")  # Replace with the path to your prediction image
            mask = sitk.ReadImage(mask_path)
            prd = sitk.ReadImage(prd_path)

            # Select the desired z-slice for both mask and prediction
            mask = sitk.GetArrayFromImage(mask[:, :, selected_slice])
            prd = sitk.GetArrayFromImage(prd[:, :, selected_slice])

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
            background_crop = background[top_h:bottom_h, left_w:right_w]

            overlay = overlay_images(mask, prd, tp_color, fp_color, fn_color, tp_alpha, fp_alpha, fn_alpha)
            overlay_list.append(overlay)

        final_overlay=sum(overlay_list)

        ax=plt.subplot(100+algosize*10+i+1)
        plt.imshow(background_crop, cmap='gray', vmin=255*background_darken_factor,vmax=255)
        plt.imshow(final_overlay, alpha=1)
        # plt.title('Custom Overlay (TP, FP, FN) on Background')
        plt.tight_layout()
        plt.axis('off')
        save_subfig(fig, ax, img_out_path, patient_id + '#' + str(selected_slice)+ '#' +algo_name + '.png')
    plt.title('Slice='+str(selected_slice))
    plt.show()
    fig.savefig(os.path.join(img_out_path,patient_id+'#'+str(selected_slice)+'.png'))
    plt.close()


if __name__ == "__main__":
    main()



