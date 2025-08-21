#!/bin/bash
# A100-Optimized Image-to-Graph Training Environment Setup Script
# Ubuntu/Linux系统专用版本 - 修复版

set -e  # 遇到错误立即停止

echo "======================================================"
echo "🚀 A100-Optimized Training Environment Setup (Ubuntu)"
echo "======================================================"
echo "开始时间: $(date)"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取当前脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$BASE_DIR/venv_a100"

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 错误处理函数
handle_error() {
    log_error "安装过程中发生错误在第 $LINENO 行，请检查日志"
    exit 1
}

# 设置错误处理
trap handle_error ERR

# 检查系统要求
check_system_requirements() {
    log_info "检查系统要求..."
    
    # 检查操作系统
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        log_success "检测到操作系统: $NAME $VERSION"
    else
        log_warning "无法检测操作系统版本，继续安装..."
    fi
    
    # 检查Python版本
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d " " -f 2)
        log_success "Python版本: $PYTHON_VERSION"
        
        # 检查Python版本是否≥3.8
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
            log_success "Python版本满足要求 (≥3.8)"
        else
            log_error "Python版本过低，需要3.8+版本"
            exit 1
        fi
    else
        log_error "Python3 未安装，请先安装Python 3.8+"
        exit 1
    fi
    
    # 检查pip
    if command -v pip3 &> /dev/null; then
        log_success "pip3 已安装"
    else
        log_error "pip3 未安装，请先安装pip3"
        exit 1
    fi
    
    # 检查GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "GPU检测失败")
        log_success "检测到GPU: $GPU_INFO"
    else
        log_warning "未检测到NVIDIA GPU或nvidia-smi，某些功能可能不可用"
    fi
    
    # 检查内存
    if command -v free &> /dev/null; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        log_info "系统内存: ${MEMORY_GB}GB"
        if [ "$MEMORY_GB" -lt 16 ]; then
            log_warning "推荐使用32GB+内存以获得最佳性能"
        fi
    fi
    
    # 检查磁盘空间
    DISK_SPACE=$(df -h "$PWD" | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "${DISK_SPACE%.*}" -lt 20 ]; then
        log_warning "磁盘可用空间不足20GB，可能影响安装"
    fi
}

# 创建虚拟环境
create_virtual_environment() {
    log_info "创建Python虚拟环境..."
    log_info "虚拟环境路径: $VENV_PATH"
    
    # 如果虚拟环境已存在，询问是否重新创建
    if [ -d "$VENV_PATH" ]; then
        log_warning "虚拟环境已存在: $VENV_PATH"
        echo -n "是否删除并重新创建? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_PATH"
            log_info "已删除旧虚拟环境"
        else
            log_info "使用现有虚拟环境"
            return 0
        fi
    fi
    
    # 创建虚拟环境
    python3 -m venv "$VENV_PATH"
    
    # 激活虚拟环境
    source "$VENV_PATH/bin/activate"
    
    # 升级pip
    python -m pip install --upgrade pip setuptools wheel
    
    log_success "虚拟环境创建完成: $VENV_PATH"
}

# 安装系统依赖
install_system_dependencies() {
    log_info "检查并安装系统依赖..."
    
    # 检查是否有sudo权限
    if command -v sudo &> /dev/null && sudo -n true 2>/dev/null; then
        log_info "检测到sudo权限，安装系统依赖..."
        
        # 更新包列表
        sudo apt-get update -qq
        
        # 安装基本依赖
        sudo apt-get install -y -qq \
            build-essential \
            python3-dev \
            python3-pip \
            libffi-dev \
            libssl-dev \
            libjpeg-dev \
            libpng-dev \
            libhdf5-dev \
            pkg-config \
            git \
            wget \
            curl
            
        log_success "系统依赖安装完成"
    else
        log_warning "无sudo权限，跳过系统依赖安装"
        log_warning "如果安装过程中出错，请手动安装build-essential python3-dev等依赖"
    fi
}

# 安装PyTorch和CUDA支持
install_pytorch() {
    log_info "安装PyTorch和CUDA支持..."
    
    # 检查CUDA版本
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
        log_info "检测到CUDA版本: $CUDA_VERSION"
        
        # 根据CUDA版本选择PyTorch
        if [[ "$CUDA_VERSION" == "11.8" ]]; then
            log_info "安装PyTorch for CUDA 11.8..."
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
        elif [[ "$CUDA_VERSION" =~ ^12\. ]]; then
            log_info "安装PyTorch for CUDA 12.1..."
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
        else
            log_warning "CUDA版本 $CUDA_VERSION，尝试安装通用CUDA版本..."
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
        fi
    else
        log_warning "未检测到CUDA，安装CPU版本PyTorch..."
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    fi
    
    log_success "PyTorch安装完成"
}

# 安装核心库
install_core_libraries() {
    log_info "安装核心库..."
    
    # 基础科学计算
    pip install numpy==1.24.3 scipy==1.11.3 pandas==2.0.3
    
    # 机器学习
    pip install scikit-learn==1.3.0 networkx==3.1
    
    # 图神经网络相关
    pip install torch-geometric==2.3.1 || log_warning "torch-geometric安装失败，将尝试简化安装"
    pip install torch-cluster torch-sparse torch-scatter || log_warning "部分torch扩展安装失败"
    
    log_success "核心库安装完成"
}

# 安装可视化库
install_visualization_libraries() {
    log_info "安装可视化库..."
    
    pip install matplotlib==3.7.2 seaborn==0.12.2 plotly==5.15.0
    pip install tqdm==4.66.1
    
    log_success "可视化库安装完成"
}

# 安装图像处理库
install_image_processing() {
    log_info "安装图像处理库..."
    
    pip install pillow==10.0.0
    pip install opencv-python==4.8.0.76 || log_warning "OpenCV安装失败，将尝试headless版本"
    pip install opencv-python-headless==4.8.0.76 || log_warning "OpenCV headless安装失败"
    
    log_success "图像处理库安装完成"
}

# 安装数据处理库
install_data_libraries() {
    log_info "安装数据处理库..."
    
    pip install h5py==3.9.0 pyyaml==6.0.1
    pip install requests==2.31.0
    
    log_success "数据处理库安装完成"
}

# 安装监控工具
install_monitoring_tools() {
    log_info "安装监控工具..."
    
    pip install psutil==5.9.5
    pip install gpustat==1.1.1 || log_warning "gpustat安装失败"
    pip install tensorboard==2.14.0
    
    log_success "监控工具安装完成"
}

# 安装开发工具
install_dev_tools() {
    log_info "安装开发工具..."
    
    pip install jupyter==1.0.0 || log_warning "Jupyter安装失败"
    pip install ipython==8.14.0
    
    log_success "开发工具安装完成"
}

# 验证安装
verify_installation() {
    log_info "验证安装..."
    
    # 创建验证脚本
    cat > "$SCRIPT_DIR/verify_install.py" << 'EOF'
#!/usr/bin/env python3
import sys
import importlib

def check_package(name, version_attr='__version__'):
    try:
        module = importlib.import_module(name)
        version = getattr(module, version_attr, 'Unknown')
        print(f"✅ {name}: {version}")
        return True
    except ImportError:
        print(f"❌ {name}: Not installed")
        return False

def main():
    print("=== 环境验证 ===")
    print(f"Python: {sys.version}")
    print()
    
    # 核心库验证
    essential_packages = [
        'torch', 'torchvision', 'numpy', 'scipy', 'pandas', 
        'sklearn', 'matplotlib', 'PIL', 'networkx'
    ]
    
    optional_packages = [
        'cv2', 'plotly', 'h5py', 'yaml', 'tqdm'
    ]
    
    print("核心库验证:")
    failed = 0
    for pkg in essential_packages:
        if not check_package(pkg):
            failed += 1
    
    print("\n可选库验证:")
    for pkg in optional_packages:
        check_package(pkg)
    
    # PyTorch验证
    try:
        import torch
        print(f"\nPyTorch环境验证:")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # 简单测试
        x = torch.randn(2, 3)
        y = torch.mm(x, x.t())
        print("✅ PyTorch基本功能测试通过")
        
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
        failed += 1
    
    print(f"\n验证完成! 关键库失败数量: {failed}")
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF
    
    # 运行验证
    python "$SCRIPT_DIR/verify_install.py"
    verification_result=$?
    
    # 清理验证脚本
    rm -f "$SCRIPT_DIR/verify_install.py"
    
    if [ $verification_result -eq 0 ]; then
        log_success "环境验证通过!"
    else
        log_warning "环境验证有警告，但可以继续使用"
    fi
}

# 创建配置文件
create_configuration() {
    log_info "创建配置文件..."
    
    # 创建配置目录
    mkdir -p "$SCRIPT_DIR/configs"
    
    # 创建环境变量脚本
    cat > "$SCRIPT_DIR/configs/env_setup.sh" << EOF
#!/bin/bash
# A100优化环境变量配置

echo "🚀 加载A100优化环境变量..."

# CUDA设置
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0

# PyTorch优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_CUDNN_V8_API_ENABLED=1

# 并行处理优化
export OMP_NUM_THREADS=\$(nproc)
export MKL_NUM_THREADS=\$(nproc)

# Python优化
export PYTHONPATH="\${PYTHONPATH}:$SCRIPT_DIR"
export PYTHONUNBUFFERED=1

# Matplotlib后端设置（避免GUI问题）
export MPLBACKEND=Agg

echo "✅ A100环境变量配置完成"
EOF
    
    chmod +x "$SCRIPT_DIR/configs/env_setup.sh"
    
    # 创建激活脚本
    cat > "$SCRIPT_DIR/activate_a100.sh" << EOF
#!/bin/bash
# A100环境激活脚本

echo "======================================================"
echo "🚀 激活A100训练环境"
echo "======================================================"

# 激活虚拟环境
echo "激活虚拟环境..."
source "$VENV_PATH/bin/activate"

# 加载环境变量
echo "加载环境变量..."
source "$SCRIPT_DIR/configs/env_setup.sh"

# 切换到工作目录
cd "$SCRIPT_DIR"

# 显示环境信息
echo ""
echo "=== 环境信息 ==="
echo "Python: \$(python --version 2>&1)"
echo "工作目录: \$(pwd)"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo '未检测到')"
fi

echo ""
echo "✅ A100环境已激活!"
echo "可用命令:"
echo "  python data/data_generation_a100.py      # 数据生成"
echo "  python training_pipeline_a100.py         # 模型训练"
echo "  python model_comparison_a100.py          # 模型比较"
echo "  python utils/visualization_a100.py       # 结果可视化"
echo ""
EOF
    
    chmod +x "$SCRIPT_DIR/activate_a100.sh"
    
    # 创建简化的requirements.txt
    cat > "$SCRIPT_DIR/requirements.txt" << 'EOF'
# A100-Optimized Requirements (Essential packages)
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
numpy==1.24.3
scipy==1.11.3
pandas==2.0.3
scikit-learn==1.3.0
networkx==3.1
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
pillow==10.0.0
opencv-python-headless==4.8.0.76
h5py==3.9.0
pyyaml==6.0.1
tqdm==4.66.1
psutil==5.9.5
tensorboard==2.14.0
jupyter==1.0.0
ipython==8.14.0
requests==2.31.0
torch-geometric==2.3.1
EOF
    
    log_success "配置文件创建完成"
}

# 主安装流程
main() {
    log_info "开始A100环境安装流程..."
    
    # 检查系统要求
    check_system_requirements
    
    # 安装系统依赖
    install_system_dependencies
    
    # 创建虚拟环境
    create_virtual_environment
    
    # 确保虚拟环境已激活
    source "$VENV_PATH/bin/activate"
    
    # 安装各类库（按重要性顺序）
    install_pytorch
    install_core_libraries
    install_visualization_libraries
    install_image_processing
    install_data_libraries
    install_monitoring_tools
    install_dev_tools
    
    # 验证安装
    verify_installation
    
    # 创建配置文件
    create_configuration
    
    # 完成信息
    echo ""
    echo "======================================================"
    log_success "🎉 A100环境安装完成!"
    echo "======================================================"
    echo ""
    echo "环境信息:"
    echo "  虚拟环境位置: $VENV_PATH"
    echo "  工作目录: $SCRIPT_DIR"
    echo "  配置文件: $SCRIPT_DIR/configs/"
    echo ""
    echo "使用方法:"
    echo "  1. 激活环境: source $SCRIPT_DIR/activate_a100.sh"
    echo "  2. 或手动激活: source $VENV_PATH/bin/activate && source $SCRIPT_DIR/configs/env_setup.sh"
    echo ""
    echo "快速开始:"
    echo "  cd $SCRIPT_DIR"
    echo "  source activate_a100.sh"
    echo "  python data/data_generation_a100.py"
    echo ""
    log_success "安装完成时间: $(date)"
}

# 运行主程序
main "$@"