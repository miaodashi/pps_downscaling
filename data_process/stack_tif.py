import os
import glob
import numpy as np
from osgeo import gdal
from tqdm import tqdm


def stack_tif_files(input_dir, output_file):
    """
    将目录中的多个单波段TIF文件叠加为一个多波段TIF文件
    
    参数:
        input_dir: 输入目录，包含要叠加的TIF文件
        output_file: 输出文件路径，保存叠加后的多波段TIF文件
    
    返回:
        成功叠加的波段数量
    """
    # 获取所有tif文件
    tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    if not tif_files:
        print(f"在 {input_dir} 中没有找到TIF文件")
        return 0
    
    print(f"找到 {len(tif_files)} 个TIF文件待叠加")
    
    # 打开第一个文件作为参考
    reference_ds = gdal.Open(tif_files[0], gdal.GA_ReadOnly)
    if reference_ds is None:
        print(f"无法打开参考文件 {tif_files[0]}")
        return 0
    
    # 获取参考文件的基本信息
    width = reference_ds.RasterXSize
    height = reference_ds.RasterYSize
    geo_transform = reference_ds.GetGeoTransform()
    projection = reference_ds.GetProjection()
    reference_ds = None
    
    # 创建输出文件，波段数等于文件数
    driver = gdal.GetDriverByName("GTiff")
    output_ds = driver.Create(output_file, width, height, len(tif_files), gdal.GDT_Float32, 
                              options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'])
    
    if output_ds is None:
        print(f"无法创建输出文件 {output_file}")
        return 0
    
    # 设置地理信息
    output_ds.SetGeoTransform(geo_transform)
    output_ds.SetProjection(projection)
    
    # 为每个输出波段设置描述（使用输入文件名）
    for i, tif_file in enumerate(tqdm(tif_files, desc="叠加TIF文件")):

        print(tif_file)
        # 读取当前文件
        src_ds = gdal.Open(tif_file, gdal.GA_ReadOnly)
        if src_ds is None:
            print(f"警告：无法打开 {tif_file}，跳过")
            continue
        
        # 检查尺寸是否一致
        if (src_ds.RasterXSize != width or src_ds.RasterYSize != height):
            print(f"警告：{tif_file} 的尺寸与参考文件不一致，跳过")
            src_ds = None
            continue
        
        # 读取数据
        src_band = src_ds.GetRasterBand(1)  # 假设输入都是单波段
        data = src_band.ReadAsArray()
        nodata_value = src_band.GetNoDataValue()
        
        # 写入输出文件
        band_idx = i + 1
        output_band = output_ds.GetRasterBand(band_idx)
        
        # 设置波段描述（使用文件名）
        band_name = os.path.basename(tif_file).split('.')[0]
        output_band.SetDescription(band_name)
        
        # 设置NoData值
        if nodata_value is not None:
            output_band.SetNoDataValue(nodata_value)
        
        # 写入数据
        output_band.WriteArray(data)
        output_band.FlushCache()
        
        # 清理
        src_ds = None
    
    # 构建金字塔（缩略图）以加快显示
    output_ds.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64])
    
    # 清理资源
    output_ds = None
    
    print(f"叠加完成，已将 {len(tif_files)} 个波段保存至: {output_file}")
    return len(tif_files)


if __name__ == "__main__":
    
    # 输出目录：用于保存归一化后的文件
    norm_dir = "D:/SoilErosion/PPS_Downscaling/500m/Environment/normalized"
    
    # 将归一化后的文件叠加为多波段TIF
    print("\n步骤2：叠加为多波段TIF")
    stacked_file = "D:/SoilErosion/PPS_Downscaling/500m/Environment/stacked_env_data.tif"
    stack_tif_files(norm_dir, stacked_file)