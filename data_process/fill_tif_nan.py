import os
import numpy as np
import argparse
import rasterio
from rasterio.fill import fillnodata
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import ndimage

warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)


def check_tif_nan(tif_path):
    """检查单个TIF文件是否包含NaN值
    
    参数:
        tif_path: TIF文件路径
    
    返回:
        包含检查结果的字典
    """
    try:
        with rasterio.open(tif_path) as src:
            # 读取数据
            data = src.read(1)  # 读取第一个波段
            
            # 检查NaN值和NoData值
            nodata = src.nodata if src.nodata is not None else np.nan
            mask = np.logical_or(np.isnan(data), data == nodata)
            has_nan = mask.any()
            
            nan_count = mask.sum() if has_nan else 0
            nan_percent = (nan_count / data.size) * 100 if has_nan else 0
            
            result = {
                'file': tif_path,
                'has_nan': has_nan,
                'nan_count': int(nan_count),
                'nan_percent': nan_percent,
                'shape': data.shape,
                'min': float(np.nanmin(data)) if not np.all(np.isnan(data)) else None,
                'max': float(np.nanmax(data)) if not np.all(np.isnan(data)) else None,
                'mean': float(np.nanmean(data)) if not np.all(np.isnan(data)) else None
            }
            
            print(f"\n检查结果: {os.path.basename(tif_path)}")
            print(f"  - 有NaN值: {'是' if has_nan else '否'}")
            if has_nan:
                print(f"  - NaN数量: {nan_count} ({nan_percent:.2f}%)")
                print(f"  - 图像形状: {data.shape}")
                print(f"  - 有效值范围: {result['min']} 至 {result['max']}")
            
            return has_nan
    
    except Exception as e:
        print(f"检查文件时出错: {e}")
        return {'file': tif_path, 'error': str(e)}


def fill_tif_nan(tif_path, output_path=None, max_search_distance=100, visualize=False):
    """使用插值方法填充TIF文件中的NaN值
    
    参数:
        tif_path: 输入TIF文件路径
        output_path: 输出TIF文件路径(如果为None则自动生成)
        max_search_distance: 插值搜索距离
        visualize: 是否可视化处理结果
        
    返回:
        处理结果信息的字典
    """
    # 如果未提供输出路径，则自动生成
    if output_path is None:
        base_path, ext = os.path.splitext(tif_path)
        output_path = f"{base_path}_filled{ext}"
    
    try:
        # 首先检查文件
        has_nan = check_tif_nan(tif_path)
        
        # 如果没有NaN值，则直接返回
        if not has_nan:
            print(f"\n文件 {os.path.basename(tif_path)} 没有NaN值，无需填充。")
            return {'has_nan': False, 'file': tif_path}
        
        print(f"\n正在处理文件: {os.path.basename(tif_path)}")
        print(f"  - 输出文件: {os.path.basename(output_path)}")
        print(f"  - 最大搜索距离: {max_search_distance}像素")
        
        # 使用rasterio打开TIF文件
        with rasterio.open(tif_path) as src:
            # 读取数据和元数据
            profile = src.profile.copy()
            data = src.read(1)  # 读取第一个波段
            
            # 创建掩码
            nodata = src.nodata if src.nodata is not None else np.nan
            mask = np.logical_or(np.isnan(data), data == nodata)
            
            # 显示处理前状态
            if visualize:
                plt.figure(figsize=(12, 5))
                plt.subplot(121)
                plt.title("处理前")
                plt.imshow(data, cmap='viridis')
                plt.colorbar(label='值')
                plt.axis('off')
            
            # 使用最近邻算法填充NaN值
            print("  - 正在进行最近邻填充...")
            
            # 复制原始数据
            filled_data = np.copy(data)
            
            # 计算每个像素到最近有效像素的距离和索引
            print("  - 计算距离变换...")
            dist, indices = ndimage.distance_transform_edt(
                mask, 
                return_distances=True, 
                return_indices=True
            )
            
            # 使用最近的有效像素值填充NaN位置
            print("  - 应用最近邻填充...")
            # 对所有NaN位置应用填充
            filled_data[mask] = data[tuple(indices[:, mask])]
            
            # 检查是否还有NaN值
            remaining_nans = np.logical_or(np.isnan(filled_data), filled_data == nodata).sum()
            if remaining_nans > 0:
                print(f"  - 警告: 填充后仍有 {remaining_nans} 个NaN值，可能需要增加搜索距离")
            else:
                print("  - 所有NaN值已成功填充")
            
            # 显示处理后状态
            if visualize:
                plt.subplot(122)
                plt.title("处理后")
                plt.imshow(filled_data, cmap='viridis')
                plt.colorbar(label='值')
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            
            # 将填充后的数据写入新文件
            print(f"  - 保存填充后的数据到: {output_path}")
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(filled_data, 1)
            
            # 获取NaN计数
            with rasterio.open(tif_path) as src_check:
                data_check = src_check.read(1)
                nodata_check = src_check.nodata if src_check.nodata is not None else np.nan
                mask_check = np.logical_or(np.isnan(data_check), data_check == nodata_check)
                original_nan_count = mask_check.sum()
                
            return {
                'input_file': tif_path,
                'output_file': output_path,
                'original_nan_count': int(original_nan_count),
                'remaining_nan_count': int(remaining_nans),
                'success': remaining_nans == 0
            }
    
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return {'input_file': tif_path, 'error': str(e)}


if __name__ == "__main__":
    input_folder = r"D:\SoilErosion\PPS_Downscaling\500m\Environment"
    
    # for file in os.listdir(input_folder):
    #     if file.endswith(".tif"):
    #         if check_tif_nan(os.path.join(input_folder, file)):
    #             fill_tif_nan(os.path.join(input_folder, file), os.path.join(input_folder, file.replace(".tif", "_filled.tif")), 100, True)


    check_tif_nan(os.path.join(input_folder, "stacked_env_data.tif"))