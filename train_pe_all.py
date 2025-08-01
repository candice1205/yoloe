from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train_pe import YOLOEPETrainer, YOLOEPESegTrainer
import os
from ultralytics.nn.tasks import guess_model_scale
from ultralytics.utils import yaml_load, LOGGER
import torch

def train_yoloe():
    # 设置Python哈希种子以确保结果可重复
    os.environ["PYTHONHASHSEED"] = "0"

    # 修改后的数据集配置文件路径
    data = r"C:\Users\Administrator\Desktop\coco8.yaml"
    # 模型配置文件路径
    model_path = "yoloe-v8s-seg.yaml"

    # 根据模型路径猜测模型规模
    scale = guess_model_scale(model_path)
    cfg_dir = "ultralytics/cfg"
    default_cfg_path = f"{cfg_dir}/default.yaml"
    extend_cfg_path = f"{cfg_dir}/coco_{scale}_train.yaml"

    # 加载默认配置和扩展配置
    defaults = yaml_load(default_cfg_path)
    extends = yaml_load(extend_cfg_path)
    assert(all(k in defaults for k in extends))
    LOGGER.info(f"Extends: {extends}")

    # 加载预训练的YOLOE模型
    model = YOLOE("yoloe-11s-seg.pt")

    # 确保为类别设置文本位置编码（PE）
    names = list(yaml_load(data)['names'].values())
    tpe = model.get_text_pe(names)
    pe_path = "coco-pe.pt"
    torch.save({"names": names, "pe": tpe}, pe_path)

    # 启用混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    # 获取所有可用的GPU设备
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        device = f"cuda:{','.join(map(str, range(num_gpus)))}"  # 使用所有GPU
        LOGGER.info(f"Using all available GPUs: {device}")
    else:
        device = "cpu"  # 如果没有GPU，使用CPU
        LOGGER.info("No GPUs found. Using CPU instead.")

    # 训练模型
    model.train(
        data=data,
        epochs=160,          # 训练轮数
        close_mosaic=10,    # 关闭mosaic数据增强策略的epoch数
        batch=64,           # 减少批量大小
        optimizer='AdamW',  # 优化器类型
        lr0=1e-3,           # 初始学习率
        warmup_bias_lr=0.0, # 权重偏置的学习率
        weight_decay=0.025, # 权重衰减
        momentum=0.9,       # 动量
        workers=4,          # 数据加载工作线程数
        device="cuda",      # 使用的设备，这里是所有GPU
        amp=True,           # 启用混合精度训练
        **extends,          # 扩展配置参数
        trainer=YOLOEPESegTrainer,
        train_pe_path=pe_path  # 训练时使用的PE路径
    )

if __name__ == '__main__':
    train_yoloe()