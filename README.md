
# Optimum Intel


[Intel Neural Compressor]

[neural-compressor]

[Intel Extension for PyTorch]

## 1. 配置Python虚拟环境

```bash
> cmd
> cd D:\devtools\PythonVenv
> python -m venv OptimumIntel
> D:\devtools\PythonVenv\OptimumIntel\Scripts\activate.bat
```

部署推理环境

安装rust https://rustup.rs/

```bash
> pip install --upgrade --upgrade-strategy eager "optimum[neural-compressor, ipex, openvino]"
> pip install --upgrade --upgrade-strategy eager "optimum[neural-compressor, openvino]"
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. 下载镜像

```bash
> export HF_ENDPOINT=https://hf-mirror.comCopy
> $env:HF_ENDPOINT = "https://hf-mirror.com"
```
