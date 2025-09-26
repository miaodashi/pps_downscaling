import os
import glob
import numpy as np
from osgeo import gdal
from tqdm import tqdm

def normalize_tif_files(input_dir, output_dir, method="minmax"):
    """
    归一化指定目录下的所有TIF文件
    
    参数:
        input_dir: 输入目录，包含要处理的TIF文件
        output_dir: 输出目录，用于保存归一化后的文件
        method: 归一化方法，可选"minmax"或"standard"
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有tif文件
    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    if not tif_files:
        print(f"在 {input_dir} 中没有找到TIF文件")
        return
    
    print(f"找到 {len(tif_files)} 个TIF文件待处理")
    
    # 处理每个文件
    for tif_file in tqdm(tif_files, desc="归一化处理"):
        filename = os.path.basename(tif_file)
        output_file = os.path.join(output_dir, filename)
        
        # 打开栅格文件
        dataset = gdal.Open(tif_file, gdal.GA_ReadOnly)
        if dataset is None:
            print(f"无法打开 {tif_file}，跳过")
            continue
            
        # 获取文件基本信息
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        bands_count = dataset.RasterCount
        geo_transform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        
        # 创建输出文件
        driver = gdal.GetDriverByName("GTiff")
        output_dataset = driver.Create(
            output_file, width, height, bands_count, gdal.GDT_Float32
        )
        output_dataset.SetGeoTransform(geo_transform)
        output_dataset.SetProjection(projection)
        
        # 处理每个波段
        for band_idx in range(1, bands_count + 1):
            band = dataset.GetRasterBand(band_idx)
            data = band.ReadAsArray()
            nodata_value = band.GetNoDataValue()
            
            # 创建掩码：非NoData值
            if nodata_value is not None:
                mask = ~np.isclose(data, nodata_value) & ~np.isnan(data)
            else:
                mask = ~np.isnan(data)
                
            # 只对有效值进行归一化
            valid_data = data[mask]
            if len(valid_data) == 0:
                print(f"警告：{filename} 的波段 {band_idx} 没有有效数据")
                normalized_data = data  # 如果没有有效数据，直接使用原始数据
            else:
                if method == "minmax":
                    # Min-Max归一化到[0,1]范围
                    min_val = np.min(valid_data)
                    max_val = np.max(valid_data)
                    
                    if max_val > min_val:
                        normalized_data = np.copy(data)
                        normalized_data[mask] = (valid_data - min_val) / (max_val - min_val)
                    else:
                        normalized_data = np.zeros_like(data)
                        normalized_data[mask] = 0.5  # 如果最大值和最小值相同，设为0.5
                    
                    # 保持NoData值
                    if nodata_value is not None:
                        normalized_data[~mask] = nodata_value
                        
                elif method == "standard":
                    # 标准化(Z-score归一化)
                    mean_val = np.mean(valid_data)
                    std_val = np.std(valid_data)
                    
                    if std_val > 0:
                        normalized_data = np.copy(data)
                        normalized_data[mask] = (valid_data - mean_val) / std_val
                    else:
                        normalized_data = np.zeros_like(data)
                        normalized_data[mask] = 0  # 如果标准差为0，设为0
                    
                    # 保持NoData值
                    if nodata_value is not None:
                        normalized_data[~mask] = nodata_value
                else:
                    raise ValueError(f"不支持的归一化方法: {method}")
            
            # 将归一化结果写入输出文件
            output_band = output_dataset.GetRasterBand(band_idx)
            if nodata_value is not None:
                output_band.SetNoDataValue(nodata_value)
            output_band.WriteArray(normalized_data)
            output_band.FlushCache()
        
        # 清理资源
        dataset = None
        output_dataset = None
    
    print(f"归一化处理完成，结果已保存至: {output_dir}")

if __name__ == "__main__":
    # 输入目录：包含要处理的TIF文件
    input_dir = "D:/SoilErosion/PPS_Downscaling/500m/Environment/tifs"
    
    # 输出目录：用于保存归一化后的文件
    output_dir = "D:/SoilErosion/PPS_Downscaling/500m/Environment/normalized"
    
    # 执行归一化处理 (可选"minmax"或"standard")
    normalize_tif_files(input_dir, output_dir, method="minmax")
