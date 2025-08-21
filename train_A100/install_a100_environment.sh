#!/bin/bash
# A100-Optimized Image-to-Graph Training Environment Setup Script
# 完整的A100环境安装与配置脚本

set -e  # 遇到错误立即停止

echo "======================================================"
echo "🚀 A100-Optimized Training Environment Setup"
echo "======================================================"
echo "开始时间: $(date)"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# 检查系统要求
check_system_requirements() {
    log_info "检查系统要求..."
    
    # 检查操作系统
    if [[ "$OSTYPE" == "darwin"* ]]; then
        log_success "检测到 macOS 系统"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_success "检测到 Linux 系统"
    else
        log_warning "未知操作系统: $OSTYPE，继续安装..."
    fi
    
    # 检查Python版本
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
        log_success "Python版本: $PYTHON_VERSION"
    else
        log_error "Python3 未安装，请先安装Python 3.9+"
        exit 1
    fi
    
    # 检查GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        log_success "检测到GPU: $GPU_INFO"
    else
        log_warning "未检测到NVIDIA GPU或nvidia-smi，某些功能可能不可用"
    fi
    
    # 检查内存
    if [[ "$OSTYPE" == "darwin"* ]]; then
        MEMORY_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
        log_info "系统内存: ${MEMORY_GB}GB"
        if [ $MEMORY_GB -lt 32 ]; then
            log_warning "推荐使用64GB+内存以获得最佳性能"
        fi
    fi
}

# 创建虚拟环境
create_virtual_environment() {
    log_info "创建Python虚拟环境..."
    
    # 检测当前目录，自动设置虚拟环境路径
    CURRENT_DIR=$(pwd)
    BASE_DIR=$(dirname "$CURRENT_DIR")
    VENV_PATH="$BASE_DIR/venv_a100"
    
    # 如果虚拟环境已存在，询问是否重新创建
    if [ -d "$VENV_PATH" ]; then
        log_warning "虚拟环境已存在: $VENV_PATH"
        read -p "是否删除并重新创建? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_PATH"
            log_info "已删除旧虚拟环境"
        else
            log_info "使用现有虚拟环境"
            return 0
        fi
    fi
    
    # 创建虚拟环境
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    
    # 升级pip
    pip install --upgrade pip setuptools wheel
    
    log_success "虚拟环境创建完成: $VENV_PATH"
}

# 安装PyTorch和CUDA支持
install_pytorch() {
    log_info "安装PyTorch和CUDA支持..."
    
    # 检查CUDA版本
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        log_info "检测到CUDA版本: $CUDA_VERSION"
        
        # 根据CUDA版本选择PyTorch
        if [[ "$CUDA_VERSION" == "11.8"* ]]; then
            log_info "安装PyTorch for CUDA 11.8..."
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
        elif [[ "$CUDA_VERSION" == "12."* ]]; then
            log_info "安装PyTorch for CUDA 12.1..."
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
        else
            log_warning "未知CUDA版本，安装默认PyTorch..."
            pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
        fi
    else
        log_warning "未检测到CUDA，安装CPU版本PyTorch..."
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    fi
    
    log_success "PyTorch安装完成"
}

# 安装核心科学计算库
install_scientific_libraries() {
    log_info "安装核心科学计算库..."
    
    # 数值计算基础库
    pip install numpy==1.24.3
    pip install scipy==1.11.3
    pip install pandas==2.0.3
    
    # 机器学习库
    pip install scikit-learn==1.3.0
    pip install networkx==3.1
    
    log_success "科学计算库安装完成"
}

# 安装深度学习扩展库
install_deep_learning_extensions() {
    log_info "安装深度学习扩展库..."
    
    # PyTorch Geometric 和相关库
    pip install torch-geometric==2.3.1
    pip install torch-cluster==1.6.1
    pip install torch-sparse==0.6.17
    pip install torch-scatter==2.1.1
    pip install torch-spline-conv==1.2.2
    
    # 图处理库
    pip install dgl==1.1.2
    
    log_success "深度学习扩展库安装完成"
}

# 安装可视化库
install_visualization_libraries() {
    log_info "安装可视化库..."
    
    # 静态图表库
    pip install matplotlib==3.7.2
    pip install seaborn==0.12.2
    
    # 交互式可视化
    pip install plotly==5.15.0
    pip install bokeh==3.2.1
    
    # Jupyter支持
    pip install jupyter==1.0.0
    pip install jupyterlab==4.0.5
    pip install ipywidgets==8.1.0
    
    log_success "可视化库安装完成"
}

# 安装图像处理库
install_image_processing_libraries() {
    log_info "安装图像处理库..."
    
    # 基础图像处理
    pip install pillow==10.0.0
    pip install opencv-python==4.8.0.76
    pip install imageio==2.31.1
    
    # 高级图像处理
    pip install scikit-image==0.21.0
    pip install albumentations==1.3.1
    
    log_success "图像处理库安装完成"
}

# 安装数据处理和存储库
install_data_libraries() {
    log_info "安装数据处理和存储库..."
    
    # 数据格式支持
    pip install h5py==3.9.0
    pip install tables==3.8.0
    pip install pyarrow==12.0.1
    
    # 配置文件处理
    pip install pyyaml==6.0.1
    pip install toml==0.10.2
    pip install configparser==6.0.0
    
    log_success "数据处理和存储库安装完成"
}

# 安装监控和调试工具
install_monitoring_tools() {
    log_info "安装监控和调试工具..."
    
    # 进度条和日志
    pip install tqdm==4.66.1
    pip install rich==13.5.2
    pip install coloredlogs==15.0.1
    
    # 性能监控
    pip install psutil==5.9.5
    pip install gpustat==1.1.1
    pip install py3nvml==0.2.7
    
    # TensorBoard
    pip install tensorboard==2.14.0
    pip install tensorboardX==2.6.2.2
    
    # 调试工具
    pip install ipdb==0.13.13
    pip install line_profiler==4.1.1
    pip install memory_profiler==0.61.0
    
    log_success "监控和调试工具安装完成"
}

# 安装A100优化库
install_a100_optimizations() {
    log_info "安装A100优化库..."
    
    # NVIDIA优化库
    if command -v nvidia-smi &> /dev/null; then
        # 尝试安装NVIDIA APEX (混合精度训练)
        log_info "尝试安装NVIDIA APEX..."
        pip install apex || log_warning "APEX安装失败，将使用PyTorch内置AMP"
        
        # NVIDIA DALI (数据加载优化)
        log_info "尝试安装NVIDIA DALI..."
        pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110 || log_warning "DALI安装失败，将使用标准数据加载"
        
        # CuPy (GPU加速数组操作)
        pip install cupy-cuda11x==12.2.0 || log_warning "CuPy安装失败"
    fi
    
    # 数值优化库
    pip install numba==0.57.1
    pip install llvmlite==0.40.1
    
    log_success "A100优化库安装完成"
}

# 安装实用工具库
install_utility_libraries() {
    log_info "安装实用工具库..."
    
    # 文件操作
    pip install pathlib2==2.3.7.post1
    pip install filelock==3.12.2
    
    # 时间处理
    pip install python-dateutil==2.8.2
    pip install pytz==2023.3
    
    # 网络请求
    pip install requests==2.31.0
    pip install urllib3==2.0.4
    
    # 多进程和并发
    pip install joblib==1.3.2
    pip install multiprocess==0.70.15
    
    # 配置管理
    pip install hydra-core==1.3.2
    pip install omegaconf==2.3.0
    
    log_success "实用工具库安装完成"
}

# 安装开发工具
install_development_tools() {
    log_info "安装开发工具..."
    
    # 代码格式化
    pip install black==23.7.0
    pip install isort==5.12.0
    pip install autopep8==2.0.2
    
    # 代码检查
    pip install flake8==6.0.0
    pip install pylint==2.17.5
    pip install mypy==1.5.1
    
    # 测试框架
    pip install pytest==7.4.0
    pip install pytest-cov==4.1.0
    pip install pytest-mock==3.11.1
    
    log_success "开发工具安装完成"
}

# 验证安装
verify_installation() {
    log_info "验证安装..."
    
    # 创建验证脚本
    cat > verify_install.py << 'EOF'
import sys
import importlib
import torch

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
    core_packages = [
        'torch', 'torchvision', 'torchaudio',
        'numpy', 'scipy', 'pandas', 'sklearn',
        'matplotlib', 'seaborn', 'plotly',
        'networkx', 'PIL', 'cv2'
    ]
    
    print("核心库验证:")
    failed = 0
    for pkg in core_packages:
        if not check_package(pkg):
            failed += 1
    
    # PyTorch验证
    print("\nPyTorch环境验证:")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # A100特性验证
    if torch.cuda.is_available():
        print("\nA100特性验证:")
        try:
            # 测试混合精度
            scaler = torch.cuda.amp.GradScaler()
            print("✅ 混合精度支持 (AMP)")
        except:
            print("❌ 混合精度支持")
        
        try:
            # 测试Tensor Core
            a = torch.randn(64, 64, dtype=torch.half).cuda()
            b = torch.randn(64, 64, dtype=torch.half).cuda()
            c = torch.mm(a, b)
            print("✅ Tensor Core支持")
        except:
            print("❌ Tensor Core支持")
    
    print(f"\n验证完成! 失败包数量: {failed}")
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF
    
    # 运行验证
    python verify_install.py
    verification_result=$?
    
    # 清理验证脚本
    rm verify_install.py
    
    if [ $verification_result -eq 0 ]; then
        log_success "环境验证通过!"
    else
        log_error "环境验证失败，请检查安装"
        return 1
    fi
}

# 创建环境变量配置
create_environment_config() {
    log_info "创建环境变量配置..."
    
    # 创建配置目录
    CURRENT_DIR=$(pwd)
    mkdir -p "$CURRENT_DIR/configs"
    
    # 创建环境变量脚本
    cat > /Users/jeremyfang/Downloads/image_to_graph/train_A100/configs/env_setup.sh << 'EOF'
#!/bin/bash
# A100优化环境变量配置

echo "🚀 加载A100优化环境变量..."

# CUDA设置
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0

# PyTorch优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# A100 Tensor Core优化
export NVIDIA_TF32_OVERRIDE=1
export CUDA_TENSOR_CORE_SPLIT_K_LIMIT=8

# 并行处理优化
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# 内存优化
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
export CUDA_CACHE_MAXSIZE=2147483648

# Python优化
export PYTHONPATH="${PYTHONPATH}:/Users/jeremyfang/Downloads/image_to_graph/train_A100"
export PYTHONUNBUFFERED=1

# Matplotlib后端设置（避免GUI问题）
export MPLBACKEND=Agg

echo "✅ A100环境变量配置完成"
EOF
    
    chmod +x /Users/jeremyfang/Downloads/image_to_graph/train_A100/configs/env_setup.sh
    
    log_success "环境变量配置创建完成"
}

# 创建启动脚本
create_activation_script() {
    log_info "创建环境激活脚本..."
    
    cat > /Users/jeremyfang/Downloads/image_to_graph/train_A100/activate_a100.sh << 'EOF'
#!/bin/bash
# A100环境激活脚本

echo "======================================================"
echo "🚀 激活A100训练环境"
echo "======================================================"

# 激活虚拟环境
echo "激活虚拟环境..."
source /Users/jeremyfang/Downloads/image_to_graph/venv_a100/bin/activate

# 加载环境变量
echo "加载环境变量..."
source /Users/jeremyfang/Downloads/image_to_graph/train_A100/configs/env_setup.sh

# 切换到工作目录
cd /Users/jeremyfang/Downloads/image_to_graph/train_A100

# 显示环境信息
echo ""
echo "=== 环境信息 ==="
echo "Python: $(python --version)"
echo "工作目录: $(pwd)"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
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
    
    chmod +x /Users/jeremyfang/Downloads/image_to_graph/train_A100/activate_a100.sh
    
    log_success "环境激活脚本创建完成"
}

# 创建requirements.txt
create_requirements_file() {
    log_info "创建requirements.txt文件..."
    
    cat > /Users/jeremyfang/Downloads/image_to_graph/train_A100/requirements.txt << 'EOF'
# A100-Optimized Image-to-Graph Training Requirements
# Generated by install_a100_environment.sh

# PyTorch and CUDA Support
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0

# Core Scientific Computing
numpy==1.24.3
scipy==1.11.3
pandas==2.0.3
scikit-learn==1.3.0
networkx==3.1

# Deep Learning Extensions
torch-geometric==2.3.1
torch-cluster==1.6.1
torch-sparse==0.6.17
torch-scatter==2.1.1
torch-spline-conv==1.2.2
dgl==1.1.2

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
bokeh==3.2.1

# Jupyter Support
jupyter==1.0.0
jupyterlab==4.0.5
ipywidgets==8.1.0

# Image Processing
pillow==10.0.0
opencv-python==4.8.0.76
imageio==2.31.1
scikit-image==0.21.0
albumentations==1.3.1

# Data Processing and Storage
h5py==3.9.0
tables==3.8.0
pyarrow==12.0.1
pyyaml==6.0.1
toml==0.10.2

# Monitoring and Debugging
tqdm==4.66.1
rich==13.5.2
coloredlogs==15.0.1
psutil==5.9.5
gpustat==1.1.1
py3nvml==0.2.7
tensorboard==2.14.0
tensorboardX==2.6.2.2
ipdb==0.13.13
line_profiler==4.1.1
memory_profiler==0.61.0

# Numerical Optimizations
numba==0.57.1
llvmlite==0.40.1
cupy-cuda11x==12.2.0

# Utilities
pathlib2==2.3.7.post1
filelock==3.12.2
python-dateutil==2.8.2
pytz==2023.3
requests==2.31.0
urllib3==2.0.4
joblib==1.3.2
multiprocess==0.70.15
hydra-core==1.3.2
omegaconf==2.3.0

# Development Tools
black==23.7.0
isort==5.12.0
autopep8==2.0.2
flake8==6.0.0
pylint==2.17.5
mypy==1.5.1
pytest==7.4.0
pytest-cov==4.1.0
pytest-mock==3.11.1
EOF
    
    log_success "requirements.txt文件创建完成"
}

# 主安装流程
main() {
    log_info "开始A100环境安装流程..."
    
    # 检查系统要求
    check_system_requirements
    
    # 创建虚拟环境
    create_virtual_environment
    
    # 激活虚拟环境
    source /Users/jeremyfang/Downloads/image_to_graph/venv_a100/bin/activate
    
    # 安装各类库
    install_pytorch
    install_scientific_libraries
    install_deep_learning_extensions
    install_visualization_libraries
    install_image_processing_libraries
    install_data_libraries
    install_monitoring_tools
    install_a100_optimizations
    install_utility_libraries
    install_development_tools
    
    # 验证安装
    verify_installation
    
    # 创建配置文件
    create_environment_config
    create_activation_script
    create_requirements_file
    
    # 完成信息
    echo ""
    echo "======================================================"
    log_success "🎉 A100环境安装完成!"
    echo "======================================================"
    echo ""
    echo "环境信息:"
    echo "  虚拟环境位置: /Users/jeremyfang/Downloads/image_to_graph/venv_a100"
    echo "  工作目录: /Users/jeremyfang/Downloads/image_to_graph/train_A100"
    echo "  配置文件: /Users/jeremyfang/Downloads/image_to_graph/train_A100/configs/"
    echo ""
    echo "使用方法:"
    echo "  1. 激活环境: source /Users/jeremyfang/Downloads/image_to_graph/train_A100/activate_a100.sh"
    echo "  2. 运行训练: python training_pipeline_a100.py"
    echo "  3. 查看帮助: python --help"
    echo ""
    echo "快速开始:"
    echo "  cd /Users/jeremyfang/Downloads/image_to_graph/train_A100"
    echo "  ./activate_a100.sh"
    echo "  python data/data_generation_a100.py"
    echo "  python training_pipeline_a100.py"
    echo ""
    log_success "安装完成时间: $(date)"
}

# 错误处理函数
handle_error() {
    log_error "安装过程中发生错误，请检查日志"
    exit 1
}

# 设置错误处理
trap handle_error ERR

# 运行主程序
main "$@"