#!/usr/bin/env python3
"""
A100混合精度训练修复脚本
修复PyTorch 2.0+版本兼容性和数据类型不匹配问题
"""

import os
import re

def fix_file(file_path, fixes):
    """应用修复到指定文件"""
    if not os.path.exists(file_path):
        print(f"⚠️  文件不存在: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    fixed = False
    
    for pattern, replacement in fixes:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
            fixed = True
    
    if fixed and content != original_content:
        # 备份原文件
        backup_path = file_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # 写入修复后的内容
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 已修复: {file_path}")
        print(f"📁 备份文件: {backup_path}")
        return True
    else:
        print(f"ℹ️  无需修复: {file_path}")
        return False

def main():
    print("🔧 A100混合精度训练修复工具")
    print("=" * 50)
    
    # 获取当前目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Model A修复
    model_a_fixes = [
        # 修复数据类型不匹配问题
        (
            r'adjacency\[b\]\[valid_indices\[:, None\], valid_indices\[None, :\]\] = edge_matrix',
            'adjacency[b][valid_indices[:, None], valid_indices[None, :]] = edge_matrix.to(adjacency.dtype)'
        ),
    ]
    
    # Model B修复 (类似问题)
    model_b_fixes = [
        # 修复可能的数据类型不匹配
        (
            r'adjacency_matrix\[b, :n_valid, :n_valid\] = similarity_matrix',
            'adjacency_matrix[b, :n_valid, :n_valid] = similarity_matrix.to(adjacency_matrix.dtype)'
        ),
    ]
    
    # 训练管道修复（已经在之前的编辑中完成，这里确保完整性）
    training_fixes = [
        # 确保import语句的兼容性处理
        (
            r'from torch\.cuda\.amp import GradScaler, autocast',
            '''# Mixed precision imports with compatibility
try:
    from torch.amp import GradScaler, autocast
    _use_new_amp_api = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _use_new_amp_api = False'''
        ),
    ]
    
    # 应用修复
    files_to_fix = [
        (os.path.join(base_dir, 'models', 'model_a_gnn_a100.py'), model_a_fixes),
        (os.path.join(base_dir, 'models', 'model_b_similarity_a100.py'), model_b_fixes),
        (os.path.join(base_dir, 'training_pipeline_a100.py'), training_fixes),
    ]
    
    fixed_count = 0
    for file_path, fixes in files_to_fix:
        if fix_file(file_path, fixes):
            fixed_count += 1
    
    print("\n" + "=" * 50)
    print(f"🎉 修复完成! 共修复 {fixed_count} 个文件")
    
    # 额外检查：确保Model B也没有类似问题
    model_b_path = os.path.join(base_dir, 'models', 'model_b_similarity_a100.py')
    if os.path.exists(model_b_path):
        print("\n🔍 检查Model B中的潜在问题...")
        with open(model_b_path, 'r') as f:
            model_b_content = f.read()
        
        # 检查是否有类似的张量赋值问题
        dtype_issues = []
        lines = model_b_content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'adjacency_matrix[' in line and '=' in line and 'similarity' in line:
                if '.to(' not in line:
                    dtype_issues.append((i, line.strip()))
        
        if dtype_issues:
            print("⚠️  发现Model B中可能的数据类型问题：")
            for line_num, line in dtype_issues:
                print(f"   第{line_num}行: {line}")
        else:
            print("✅ Model B检查完成，未发现问题")
    
    print("\n📋 修复内容总结：")
    print("1. ✅ 修复混合精度训练中的数据类型不匹配")
    print("2. ✅ 更新PyTorch 2.0+ API兼容性")
    print("3. ✅ 确保张量操作的类型安全")
    print("\n🚀 现在可以重新运行训练:")
    print("   python training_pipeline_a100.py")

if __name__ == "__main__":
    main()