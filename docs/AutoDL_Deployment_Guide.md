# AutoDL 云端部署与训练指南 (OPPO 面试项目)

本指南旨在指导你如何将本项目中的核心模型训练代码部署到 AutoDL 的 RTX 4090 实例上，从而完成真实的模型训练过程。

## 一、 环境准备 (AutoDL 控制台)

1. **租用实例**：
   - 登录 AutoDL (https://www.autodl.com/)
   - 选择按量计费，挑选一台包含 `RTX 4090 (24GB)` 的实例。
   - **镜像选择 (非常重要)**：选择官方镜像中的 `PyTorch` 镜像。推荐版本：`PyTorch 2.1.0` + `Python 3.10` + `CUDA 11.8`。这能为你省去所有安装 torch 和 GPU 驱动的麻烦。

2. **连接实例**：
   - 实例开机后，复制 SSH 登录指令。
   - 推荐使用 VS Code 的 `Remote-SSH` 插件连接到 AutoDL 服务器。

## 二、 上传代码与数据

将你在本地开发的代码同步到 AutoDL。

1. **创建工作目录**：
   在 AutoDL 终端执行：
   ```bash
   mkdir -p /root/autodl-tmp/oppo_rec_project
   cd /root/autodl-tmp/oppo_rec_project
   ```
   *(注：强烈建议把项目放在 `autodl-tmp` 目录下，因为这个目录的数据盘容量大，适合放数据集)*

2. **上传文件**：
   将本地的 `src/mmoe_model.py` 和后续可能下载的大型数据集（如 Ali-CCP）上传到该目录。

## 三、 安装额外依赖

AutoDL 的 PyTorch 镜像已经自带了 `torch`、`numpy`、`pandas` 等绝大多数包。你只需要安装少量的额外包：

```bash
pip install scikit-learn tqdm faiss-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```
*(注：`faiss-gpu` 用于后续如果你想在云端实现高速向量检索召回)*

## 四、 运行模型训练

在 AutoDL 终端直接运行：
```bash
python src/mmoe_model.py
```

### 💡 进阶：如何把这个跑通变成“面试吹牛资本”？
当你在 AutoDL 跑模型时，不要只是跑一个脚本，建议你做以下“小动作”：

1. **保存 TensorBoard 日志**：在代码里加上 `torch.utils.tensorboard`。面试时可以说：“我监控了 MMoE 中不同 Expert 门控网络（Gate）在训练初期的震荡情况，并使用了动态学习率调整”。
2. **记录显存占用**：使用 `watch -n 1 nvidia-smi` 观察显存。面试时可以说：“在 4090 上，我发现 Batch Size 设置为 4096 时显存利用率最高（约 18GB），训练吞吐量达到了最佳的 X samples/second”。
3. **混合精度训练 (AMP)**：在代码里加上 `torch.cuda.amp.autocast()`。面试时说：“为了最大化利用 4090 的 Tensor Core 算力，我使用了 FP16 混合精度训练，将训练时间缩短了 40%，且离线 AUC 几乎无损”。

---
*(下一部分：OPPO 面试简历包装与 PPT 思路)*