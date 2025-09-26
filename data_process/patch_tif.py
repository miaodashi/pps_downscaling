from ntpath import exists
import os
import re
import glob
from shlex import join
import numpy as np
from osgeo import gdal
from tqdm import tqdm
import shutil

def clip_tif_to_npy(input_tif, output_dir, patch_size=50, overlap=0, valid_threshold=0.8):
    """
    将多波段TIF文件裁剪成小块并保存为npy格式
    
    参数:
        input_tif: 输入的多波段TIF文件路径
        output_dir: 输出目录，用于保存npy格式的数据块
        patch_size: 数据块的大小（像素数）
        overlap: 相邻块之间的重叠像素数
        valid_threshold: 有效数据的最小比例阈值（0-1之间），低于此阈值的块将被丢弃
    
    返回:
        保存的数据块数量
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开TIF文件
    dataset = gdal.Open(input_tif, gdal.GA_ReadOnly)
    if dataset is None:
        print(f"无法打开输入文件: {input_tif}")
        return 0
    
    # 获取文件基本信息
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands_count = dataset.RasterCount
    
    print(f"输入文件信息: 宽度={width}, 高度={height}, 波段数={bands_count}")
    
    # 计算步长和块数
    step_size = patch_size - overlap
    
    # 计算在x和y方向上的块数
    num_patches_x = (width - patch_size) // step_size + 1
    num_patches_y = (height - patch_size) // step_size + 1
    
    # 如果有剩余空间，添加一个额外的块
    if width - (num_patches_x * step_size + overlap) >= patch_size - overlap:
        num_patches_x += 1
    if height - (num_patches_y * step_size + overlap) >= patch_size - overlap:
        num_patches_y += 1
    
    print(f"将提取 {num_patches_x}x{num_patches_y} = {num_patches_x * num_patches_y} 个 {patch_size}x{patch_size} 大小的块")
    
    # 获取所有波段的NoData值
    nodata_values = []
    for b in range(1, bands_count + 1):
        band = dataset.GetRasterBand(b)
        nodata_value = band.GetNoDataValue()
        nodata_values.append(nodata_value)
    
    # 开始提取数据块
    successful_patches = 0
    
    for y_idx in range(num_patches_y):
        y_offset = y_idx * step_size
        # 确保不超出图像边界
        if y_offset + patch_size > height:
            y_offset = height - patch_size
        
        for x_idx in range(num_patches_x):
            x_offset = x_idx * step_size
            # 确保不超出图像边界
            if x_offset + patch_size > width:
                x_offset = width - patch_size
            
            # 创建一个数组来存储所有波段的数据
            patch_data = np.zeros((bands_count, patch_size, patch_size), dtype=np.float32)
            valid_patch = True
            
            # 读取每个波段的数据
            for b in range(1, bands_count + 1):
                band = dataset.GetRasterBand(b)
                data = band.ReadAsArray(x_offset, y_offset, patch_size, patch_size)
                
                # 检查NoData值
                if nodata_values[b-1] is not None:
                    mask = np.isclose(data, nodata_values[b-1]) | np.isnan(data)
                    valid_ratio = np.sum(~mask) / mask.size
                    if valid_ratio < valid_threshold:
                        valid_patch = False
                        break
                
                patch_data[b-1] = data
            
            # 如果是有效的数据块，则保存
            if valid_patch:
                output_file = os.path.join(output_dir, f"patch_{y_idx}_{x_idx}.npy")
                np.save(output_file, patch_data)
                successful_patches += 1
    
    dataset = None
    print(f"成功提取并保存了 {successful_patches} 个有效数据块到 {output_dir}")
    return successful_patches


def merge_npy_to_tif(npy_dir, output_tif, reference_tif, patch_size=50, overlap=0):
    """
    将npy格式的数据块拼接成一个完整的TIF文件
    
    参数:
        npy_dir: 包含npy文件的目录
        output_tif: 输出的TIF文件路径
        reference_tif: 参考TIF文件，用于获取地理信息
        patch_size: 数据块的大小（像素数）
        overlap: 相邻块之间的重叠像素数
    
    返回:
        是否成功拼接
    """
    # 获取所有npy文件
    npy_files = glob.glob(os.path.join(npy_dir, "*.npy"))
    if not npy_files:
        print(f"在 {npy_dir} 中没有找到npy文件")
        return False
    
    # 打开参考TIF文件以获取地理信息
    ref_dataset = gdal.Open(reference_tif, gdal.GA_ReadOnly)
    if ref_dataset is None:
        print(f"无法打开参考文件: {reference_tif}")
        return False
    
    # 获取参考文件的地理信息
    width = ref_dataset.RasterXSize
    height = ref_dataset.RasterYSize
    geo_transform = ref_dataset.GetGeoTransform()
    projection = ref_dataset.GetProjection()
    
    # 从第一个npy文件中获取波段数
    first_patch = np.load(npy_files[0])
    bands_count = first_patch.shape[0]
    
    print(f"创建输出TIF文件: 宽度={width}, 高度={height}, 波段数={bands_count}")
    
    # 创建输出文件
    driver = gdal.GetDriverByName("GTiff")
    output_ds = driver.Create(
        output_tif, width, height, bands_count, gdal.GDT_Float32,
        options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']
    )
    
    if output_ds is None:
        print(f"无法创建输出文件: {output_tif}")
        return False
    
    # 设置地理信息
    output_ds.SetGeoTransform(geo_transform)
    output_ds.SetProjection(projection)
    
    # 初始化结果数组(可能很大，不直接在内存中创建)
    # 我们将直接写入GDAL数据集
    
    # 获取波段描述信息
    band_descriptions = []
    for b in range(1, bands_count + 1):
        ref_band = ref_dataset.GetRasterBand(b)
        desc = ref_band.GetDescription()
        if desc:
            band_descriptions.append(desc)
        else:
            band_descriptions.append(f"Band_{b}")
    
    # 创建一个计数数组，用于记录每个像素被多少个patch覆盖
    count_array = np.zeros((height, width), dtype=np.int16)
    
    # 创建输出波段并应用描述
    output_bands = []
    for b in range(1, bands_count + 1):
        band = output_ds.GetRasterBand(b)
        if b-1 < len(band_descriptions):
            band.SetDescription(band_descriptions[b-1])
        output_bands.append(band)
        # 初始化波段数据为0
        band.Fill(0)
        band.FlushCache()
    
    # 使用正则表达式从文件名提取y_idx和x_idx
    pattern = r'patch_([0-9]+)_([0-9]+)\.npy'
    
    # 读取所有patch并放入正确的位置
    step_size = patch_size - overlap
    
    print(f"正在拼接 {len(npy_files)} 个patch...")
    for npy_file in tqdm(npy_files, desc="拼接npy文件"):
        # 从文件名中提取索引
        match = re.search(pattern, os.path.basename(npy_file))
        if not match:
            print(f"警告：无法从 {npy_file} 提取索引信息，跳过")
            continue
        
        y_idx = int(match.group(1))
        x_idx = int(match.group(2))
        
        # 计算在原图中的偏移量
        y_offset = y_idx * step_size
        x_offset = x_idx * step_size
        
        # 确保不超出图像边界
        if y_offset + patch_size > height:
            y_offset = height - patch_size
        if x_offset + patch_size > width:
            x_offset = width - patch_size
        
        # 读取patch数据
        patch_data = np.load(npy_file)
        
        # 对每个波段，将patch数据添加到对应位置
        for b in range(bands_count):
            # 读取当前波段在这个区域的数据
            band_data = output_bands[b].ReadAsArray(x_offset, y_offset, patch_size, patch_size)
            
            # 添加patch数据（这里用累加，之后会除以计数得到平均值）
            band_data = band_data + patch_data[b]
            
            # 写回波段
            output_bands[b].WriteArray(band_data, x_offset, y_offset)
        
        # 更新计数数组
        count_area = count_array[y_offset:y_offset+patch_size, x_offset:x_offset+patch_size]
        count_area += 1
        count_array[y_offset:y_offset+patch_size, x_offset:x_offset+patch_size] = count_area
    
    # 处理重叠区域：用累加的值除以计数得到平均值
    print("处理重叠区域...")
    for b in range(bands_count):
        # 对整个波段进行操作
        band_data = output_bands[b].ReadAsArray(0, 0, width, height)
        
        # 防止除以零错误
        valid_mask = count_array > 0
        band_data[valid_mask] = band_data[valid_mask] / count_array[valid_mask]
        
        # 写回波段
        output_bands[b].WriteArray(band_data)
        output_bands[b].FlushCache()
    
    # 构建金字塔以加快显示
    output_ds.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64])
    
    # 清理资源
    ref_dataset = None
    output_ds = None
    
    print(f"拼接完成，结果已保存至: {output_tif}")
    return True


if __name__ == "__main__":
    # 裁剪环境数据
    # input_tif = "D:/SoilErosion/PPS_Downscaling/500m/Environment/stacked_env_data.tif"
    # output_dir = "D:/SoilErosion/PPS_Downscaling/500m/Environment/npy_patches"

    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    #     print(f"已删除文件夹: {output_dir}")
    # os.mkdir(output_dir)
    
    # # 将TIF文件裁剪成64x64的块，每个块之间重叠32个像素
    # clip_tif_to_npy(input_tif, output_dir, patch_size=64, overlap=32, valid_threshold=0.8)


    # 裁剪土壤性质数据
    soil_tif_folder = r"D:\SoilErosion\PPS_Downscaling\500m\Soil"
    npy_path = os.path.join(soil_tif_folder, "npy_patches")
    # 如果文件夹存在，则删除
    if os.path.exists(npy_path):
        shutil.rmtree(npy_path)
        print(f"已删除文件夹: {npy_path}")
    os.mkdir(npy_path)
    
    soil_name = ["Fenli", "Nianli", "Shali", "SOC"]
    for i in soil_name:
        soil_tif = os.path.join(soil_tif_folder, i + "_hubei_bound_500m.tif")
        soil_output_dir = os.path.join(npy_path, i)
        # 如果文件夹存在，则删除
        if os.path.exists(soil_output_dir):
            shutil.rmtree(soil_output_dir)
            print(f"已删除文件夹: {soil_output_dir}")
        os.mkdir(soil_output_dir)
        clip_tif_to_npy(soil_tif, soil_output_dir, patch_size=64, overlap=32, valid_threshold=0.8)
    
    
    # 将npy文件拼接回TIF文件
    # output_tif = "D:/SoilErosion/PPS_Downscaling/500m/Environment/reconstructed_data.tif"
    # merge_npy_to_tif(output_dir, output_tif, input_tif, patch_size=64, overlap=32)
