#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSCAN网络架构可视化
绘制Multi-Scale Convolutional Attention Network的详细结构图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False

def create_mscan_architecture():
    """创建MSCAN架构图"""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # 颜色配置
    colors = {
        'input': '#E8F4FD',
        'stem': '#FFE6CC', 
        'stage': '#D4EDDA',
        'attention': '#F8D7DA',
        'fusion': '#FFF3CD',
        'output': '#E2E3E5'
    }
    
    # 1. 输入层
    input_box = FancyBboxPatch(
        (1, 12), 3, 1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['input'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(input_box)
    ax.text(2.5, 12.75, '输入: 8通道栅格数据\n[B, 8, 64, 64]\n(DEM,气温,降水,Slope,\n地表温度,B5/B7,NDVI,NDWI)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 2. StemConv层
    stem_box = FancyBboxPatch(
        (6, 12), 3, 1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['stem'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(stem_box)
    ax.text(7.5, 12.75, 'StemConv\n两个3×3卷积\n步长2+2\n[B, 64, 16, 16]', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 3. 四个Stage
    stage_info = [
        {'pos': (1, 9.5), 'name': 'Stage 1', 'depth': '3层', 'channels': 64, 'size': '16×16', 'kernel': '5×5'},
        {'pos': (6, 9.5), 'name': 'Stage 2', 'depth': '4层', 'channels': 128, 'size': '8×8', 'kernel': '7×7'},
        {'pos': (11, 9.5), 'name': 'Stage 3', 'depth': '6层', 'channels': 256, 'size': '4×4', 'kernel': '11×11'},
        {'pos': (16, 9.5), 'name': 'Stage 4', 'depth': '3层', 'channels': 512, 'size': '2×2', 'kernel': '21×21'}
    ]
    
    stage_boxes = []
    for i, info in enumerate(stage_info):
        # Stage主框
        stage_box = FancyBboxPatch(
            info['pos'], 3.5, 2,
            boxstyle="round,pad=0.1",
            facecolor=colors['stage'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(stage_box)
        stage_boxes.append(stage_box)
        
        # Stage标题
        ax.text(info['pos'][0] + 1.75, info['pos'][1] + 1.7, info['name'], 
                ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Stage详情
        ax.text(info['pos'][0] + 1.75, info['pos'][1] + 1, 
                f"{info['depth']} MSCABlock\n{info['channels']}通道\n{info['size']}\n注意力核:{info['kernel']}", 
                ha='center', va='center', fontsize=9)
    
    # 4. MSCA注意力模块详细结构 (放在右侧)
    attention_box = FancyBboxPatch(
        (1, 6), 8, 2.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['attention'],
        edgecolor='red',
        linewidth=2
    )
    ax.add_patch(attention_box)
    ax.text(5, 7.8, 'MSCA注意力模块 (Multi-Scale Convolutional Attention)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # MSCA内部结构
    msca_components = [
        {'pos': (1.5, 6.3), 'text': '5×5\n深度卷积', 'size': (1.2, 0.8)},
        {'pos': (3, 6.3), 'text': '1×7 + 7×1\n条带卷积', 'size': (1.2, 0.8)},
        {'pos': (4.5, 6.3), 'text': '1×11 + 11×1\n条带卷积', 'size': (1.2, 0.8)},
        {'pos': (6, 6.3), 'text': '1×21 + 21×1\n条带卷积', 'size': (1.2, 0.8)},
        {'pos': (7.5, 6.3), 'text': '1×1\n通道混合', 'size': (1.2, 0.8)}
    ]
    
    for comp in msca_components:
        comp_box = FancyBboxPatch(
            comp['pos'], comp['size'][0], comp['size'][1],
            boxstyle="round,pad=0.05",
            facecolor='white',
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(comp_box)
        ax.text(comp['pos'][0] + comp['size'][0]/2, comp['pos'][1] + comp['size'][1]/2, 
                comp['text'], ha='center', va='center', fontsize=8)
    
    # 5. 多尺度特征融合模块
    fusion_box = FancyBboxPatch(
        (11, 6), 7, 2.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['fusion'],
        edgecolor='orange',
        linewidth=2
    )
    ax.add_patch(fusion_box)
    ax.text(14.5, 7.8, '多尺度特征融合模块', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 融合模块内部
    fusion_steps = [
        '1. 通道转换 → 256通道',
        '2. 上采样 → 统一分辨率', 
        '3. 多尺度通道注意力',
        '4. 特征拼接与融合'
    ]
    
    for i, step in enumerate(fusion_steps):
        ax.text(11.5, 7.4 - i*0.3, step, ha='left', va='center', fontsize=9)
    
    # 6. 回归输出
    output_box = FancyBboxPatch(
        (8.5, 3), 3, 1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['output'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(output_box)
    ax.text(10, 3.75, '回归输出\n[B, 1, H, W]\n土壤属性预测', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 连接箭头
    arrows = [
        # 输入到StemConv
        {'start': (4, 12.75), 'end': (6, 12.75)},
        # StemConv到Stage1
        {'start': (7.5, 12), 'end': (2.75, 11.5)},
        # Stage之间的连接
        {'start': (4.5, 10.5), 'end': (6, 10.5)},
        {'start': (9.5, 10.5), 'end': (11, 10.5)},
        {'start': (14.5, 10.5), 'end': (16, 10.5)},
        # Stage到融合模块
        {'start': (2.75, 9.5), 'end': (12, 8.5)},
        {'start': (7.75, 9.5), 'end': (13, 8.5)},
        {'start': (12.75, 9.5), 'end': (14, 8.5)},
        {'start': (17.75, 9.5), 'end': (16, 8.5)},
        # 融合到输出
        {'start': (14.5, 6), 'end': (10, 4.5)}
    ]
    
    for arrow in arrows:
        ax.annotate('', xy=arrow['end'], xytext=arrow['start'],
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # 7. 添加特征图尺寸标注
    sizes = ['64×64', '16×16', '8×8', '4×4', '2×2']
    positions = [(2.5, 11.8), (2.75, 9), (7.75, 9), (12.75, 9), (17.75, 9)]
    
    for size, pos in zip(sizes, positions):
        ax.text(pos[0], pos[1], size, ha='center', va='center', 
                fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue'))
    
    # 8. 标题和说明
    ax.text(10, 13.5, 'MSCAN (Multi-Scale Convolutional Attention Network) 架构图', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 图例
    legend_elements = [
        ('输入层', colors['input']),
        ('预处理', colors['stem']),
        ('特征提取', colors['stage']),
        ('注意力机制', colors['attention']),
        ('特征融合', colors['fusion']),
        ('输出层', colors['output'])
    ]
    
    for i, (label, color) in enumerate(legend_elements):
        legend_box = FancyBboxPatch(
            (1 + i*3, 0.5), 2.5, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(legend_box)
        ax.text(2.25 + i*3, 0.9, label, ha='center', va='center', fontsize=9)
    
    # 关键特性说明
    features_text = """
    关键特性:
    • 多尺度感受野: 5×5, 7×7, 11×11, 21×21
    • 条带卷积: 降低计算复杂度
    • 层次化特征提取: 从局部到全局
    • 通道注意力: 自适应特征选择
    • 残差连接: 保持梯度流动
    """
    
    ax.text(19, 2, features_text, ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_msca_detail():
    """创建MSCA模块详细结构图"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(8, 9.5, 'MSCA (Multi-Scale Convolutional Attention) 详细结构', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # 输入
    input_box = FancyBboxPatch(
        (7, 8), 2, 0.8,
        boxstyle="round,pad=0.1",
        facecolor='lightblue',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(input_box)
    ax.text(8, 8.4, '输入特征\n[B, C, H, W]', ha='center', va='center', fontsize=10)
    
    # 多尺度卷积分支
    branches = [
        {'pos': (1, 6), 'name': '5×5 深度卷积', 'color': '#FFE6CC'},
        {'pos': (4, 6), 'name': '1×7 + 7×1\n条带卷积', 'color': '#D4EDDA'},
        {'pos': (7, 6), 'name': '1×11 + 11×1\n条带卷积', 'color': '#F8D7DA'},
        {'pos': (10, 6), 'name': '1×21 + 21×1\n条带卷积', 'color': '#FFF3CD'},
        {'pos': (13, 6), 'name': '1×1 通道混合', 'color': '#E2E3E5'}
    ]
    
    branch_boxes = []
    for branch in branches:
        box = FancyBboxPatch(
            branch['pos'], 2.5, 1.2,
            boxstyle="round,pad=0.1",
            facecolor=branch['color'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(box)
        branch_boxes.append(box)
        ax.text(branch['pos'][0] + 1.25, branch['pos'][1] + 0.6, 
                branch['name'], ha='center', va='center', fontsize=9)
    
    # 特征融合
    fusion_box = FancyBboxPatch(
        (6, 4), 4, 1,
        boxstyle="round,pad=0.1",
        facecolor='lightcoral',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(fusion_box)
    ax.text(8, 4.5, '特征融合 (Element-wise Add)', ha='center', va='center', fontsize=11)
    
    # 注意力权重生成
    attention_box = FancyBboxPatch(
        (6, 2), 4, 1,
        boxstyle="round,pad=0.1",
        facecolor='gold',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(attention_box)
    ax.text(8, 2.5, '注意力权重 × 输入特征', ha='center', va='center', fontsize=11)
    
    # 连接箭头
    # 输入到各分支
    for i, branch in enumerate(branches):
        start_x = 8
        end_x = branch['pos'][0] + 1.25
        ax.annotate('', xy=(end_x, 7.2), xytext=(start_x, 8),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))
    
    # 分支到融合
    for branch in branches:
        start_x = branch['pos'][0] + 1.25
        ax.annotate('', xy=(8, 5), xytext=(start_x, 6),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='green'))
    
    # 融合到注意力
    ax.annotate('', xy=(8, 3), xytext=(8, 4),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # 添加感受野大小标注
    receptive_fields = ['5×5', '7×7', '11×11', '21×21', '1×1']
    for i, (branch, rf) in enumerate(zip(branches, receptive_fields)):
        ax.text(branch['pos'][0] + 1.25, branch['pos'][1] - 0.3, 
                f'感受野: {rf}', ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # 创建MSCAN整体架构图
    print("正在生成MSCAN网络架构图...")
    fig1 = create_mscan_architecture()
    fig1.savefig('MSCAN_Architecture.png', dpi=300, bbox_inches='tight')
    print("MSCAN架构图已保存为: MSCAN_Architecture.png")
    
    # 创建MSCA模块详细图
    print("正在生成MSCA模块详细图...")
    fig2 = create_msca_detail()
    fig2.savefig('MSCA_Detail.png', dpi=300, bbox_inches='tight')
    print("MSCA详细图已保存为: MSCA_Detail.png")
    
    plt.show()
