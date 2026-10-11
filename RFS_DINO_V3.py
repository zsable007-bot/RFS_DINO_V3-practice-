"""RFS_DINO_V3.ipynb 的训练流程转写：DINOv3 冻结骨干 + 三分类头（2FSK/CW/LFM）。

与 notebook 保持一致：
    - 数据：fsdataset/{train,val,test}（ImageFolder，类别序 ['2FSK','CW','LFM']）
    - 变换：Resize(256×256) → ToTensor → Normalize（无增强，训练/验证/测试相同）
    - 模型：vit_small_patch16_dinov3.lvd1689m 冻结骨干（384 维特征）
            + Linear(384,256) → ReLU → Dropout(0.1) → Linear(256,3)
    - 训练：Adam(lr=1e-3)、CrossEntropyLoss、batch=2、10 epochs、按验证准确率选最优
    - 随机种子：42（模型初始化、数据顺序、激活）
额外（不影响训练结果）：
    - run 目录内保存 training_summary.json：逐轮曲线、测试集结果、数据统计、环境信息。

运行环境：
    F:\\Program\\Miniforge3\\envs\\dinov3_11\\python.exe RFS_DINO_V3.py
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

PROJECT_DIR = Path(__file__).resolve().parent

for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"
SEED = 42


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "fsdataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "runs")
    return parser


def make_rf_transform(config):
    height, width = config["input_size"][-2:]
    return transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["mean"], std=config["std"]),
        ]
    )


def make_head(feature_dim, num_classes):
    return nn.Sequential(
        nn.Linear(feature_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, num_classes),
    )


@torch.no_grad()
def evaluate(data_loader, backbone, classifier, criterion, device):
    backbone.eval()
    classifier.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        features = backbone(images)
        logits = classifier(features)
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size
    return loss_sum / total, 100 * correct / total


def main(argv=None):
    args = build_parser().parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Python：", sys.executable)
    print("PyTorch：", torch.__version__)
    print("timm：", timm.__version__)
    print("计算设备：", device)
    if device.type == "cuda":
        print("GPU：", torch.cuda.get_device_name(0))

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"没有找到数据集，请检查：{data_dir}")

    backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    backbone = backbone.to(device)
    backbone.requires_grad_(False)
    backbone.eval()
    trainable_count = sum(
        p.numel() for p in backbone.parameters() if p.requires_grad
    )
    print("骨干可训练参数数量：", trainable_count)

    feature_dim = backbone.num_features
    data_config = timm.data.resolve_model_data_config(backbone)
    print("特征维度：", feature_dim)
    print("输入尺寸：", data_config["input_size"])
    print("均值：", data_config["mean"])
    print("标准差：", data_config["std"])

    train_transform = make_rf_transform(data_config)
    eval_transform = make_rf_transform(data_config)

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=eval_transform)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print("类别名称：", class_names)
    print("类别编号：", train_dataset.class_to_idx)
    for name, dataset in (
        ("train", train_dataset),
        ("val", val_dataset),
        ("test", test_dataset),
    ):
        print(name, "图片数量：", len(dataset))
    assert (
        train_dataset.class_to_idx
        == val_dataset.class_to_idx
        == test_dataset.class_to_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    torch.manual_seed(args.seed)
    head = make_head(feature_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)

    baseline_loss, baseline_acc = evaluate(
        val_loader, backbone, head, criterion, device
    )
    print(f"训练前验证损失：{baseline_loss:.4f}")
    print(f"训练前验证准确率：{baseline_acc:.2f}%")

    def train_one_batch(images, labels):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            features = backbone(images)
        logits = head(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        correct = (logits.argmax(dim=1) == labels).sum().item()
        return loss.item(), correct, labels.size(0)

    def train_one_epoch():
        backbone.eval()
        head.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            batch_loss, batch_correct, batch_size = train_one_batch(images, labels)
            loss_sum += batch_loss * batch_size
            correct += batch_correct
            total += batch_size
        return loss_sum / total, 100 * correct / total

    history = []
    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = evaluate(val_loader, backbone, head, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in head.state_dict().items()
            }
        print(
            f"第 {epoch:02d} 轮 | "
            f"训练损失 {train_loss:.4f} | "
            f"训练准确率 {train_acc:.2f}% | "
            f"验证准确率 {val_acc:.2f}%"
        )

    head.load_state_dict(best_state)
    head.eval()
    print("选用轮次：", best_epoch)
    print(f"最佳验证准确率：{best_val_acc:.2f}%")
    print(f"训练前验证准确率：{baseline_acc:.2f}%")

    test_loss, test_acc = evaluate(test_loader, backbone, head, criterion, device)
    print(f"测试集损失：{test_loss:.4f}")
    print(f"测试集准确率：{test_acc:.2f}%")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    run_dir = (args.out_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_dir / "best_head.pth"
    torch.save(
        {
            "classifier_state_dict": best_state,
            "feature_dim": feature_dim,
            "num_classes": num_classes,
            "class_to_idx": train_dataset.class_to_idx,
            "model_name": MODEL_NAME,
            "data_config": data_config,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
        },
        checkpoint_path,
    )
    (run_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "model_name": MODEL_NAME,
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "data_dir": str(data_dir),
                "class_to_idx": train_dataset.class_to_idx,
                "dataset_sizes": {
                    "train": len(train_dataset),
                    "val": len(val_dataset),
                    "test": len(test_dataset),
                },
                "baseline_val_acc": baseline_acc,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "history": history,
                "torch_version": torch.__version__,
                "timm_version": timm.__version__,
                "device": str(device),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("权重保存位置：", checkpoint_path)
    print("训练摘要：", run_dir / "training_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
