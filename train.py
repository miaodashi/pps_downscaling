import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # tensorboard --logdir=runs

from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

from metrix_utils import eval_metric

from mscan import MCA_STM as MyModel
from dataset import Dataset_Train as MyDataset


def parse_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        "--datadir", type=str, default="./data/train/test_region/5", help="Data dir"
    )
    parser.add_argument(
        "--labeldir",
        type=str,
        default="",
        help="Label dir",
    )

    # training
    parser.add_argument(
        "--epoch", type=int, default=1000, help="Epoch to run [default: 1000]"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch Size during training [default: 16]",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate [default: 0.001]"
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.5, help="Dropout_Rate [default: 0.5]"
    )

    parser.add_argument("--train_rate", type=float, default=0.7, help="train rate")
    parser.add_argument("--val_rate", type=float, default=0.3, help="validate rate")

    parser.add_argument(
        "--optim",
        type=str,
        default="AdamW",
        help="Choose an optimizer [default: AdamW]",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ReduceLROnPlateau",
        help="Choose a scheduler [default: ReduceLROnPlateau]",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight Decay [default: 0.001]",
    )
    parser.add_argument(
        "--early_stop", type=int, default=50, help="Early stop [default: 50]"
    )

    parser.add_argument(
        "--beta", type=float, default=0.5, help="SmoothL1Loss beta [default: 0.5]"
    )

    # warmup
    parser.add_argument(
        "--warmup_epoch", type=int, default=10, help="Warmup [default: 10]"
    )
    parser.add_argument(
        "--warmup_lr_init",
        type=float,
        default=0.00001,
        help="Warmup learning init rate [default: 0.00001]",
    )
    parser.add_argument(
        "--warmup_lr_end",
        type=float,
        default=0.001,
        help="Warmup learning end rate [default: 0.001]",
    )

    # others
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use[default: 0]")
    parser.add_argument("--seed", type=int, default=3, help="Random seed [default: 3]")
    parser.add_argument("--label_name", type=str, default="Al2O3", help="Label name")
    parser.add_argument(
        "--timeflag",
        type=str,
        default="00000000-000000",
        help="Time flag [default: 00000000-000000]",
    )

    option = parser.parse_args()

    return option


def val_test(model, dataloader, opt) -> dict:
    """validation test

    Args:
        model : input model
        dataloader : input dataloader
        opt : options

    Returns:
        _type_: loss, r2, mae, mse, rmse, outputs, labels
    """
    model.eval()
    total_loss = []
    outputs = []
    labels = []
    outputs_ = []
    labels_ = []

    for j, (data, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Vali"):
        data = data.cuda(opt.gpu)
        label = label.cuda(opt.gpu)

        output = model(data)
        # Loss
        loss = nn.MSELoss()(output, label)
        # loss = nn.SmoothL1Loss(beta=opt.beta)(output, label)
        total_loss.append(loss.item())
        
        outputs_.append(output.detach()) 
        labels_.append(label.detach())  
            
        if (j+1) % 10 == 0:
            outputs.append(torch.cat(outputs_, dim=0).cpu())
            labels.append(torch.cat(labels_, dim=0).cpu())
            del outputs_, labels_
            torch.cuda.empty_cache()
            outputs_ = []
            labels_ = []

        # -------------------------Metric-------------------------
    if outputs_:
        outputs.append(torch.cat(outputs_, dim=0).cpu())
        labels.append(torch.cat(labels_, dim=0).cpu())
        del outputs_, labels_
        
    outputs = torch.cat(outputs, dim=0).numpy().flatten()
    labels = torch.cat(labels, dim=0).numpy().flatten()  

    metric_dict = eval_metric(outputs, labels)
    metric_dict["loss"] = np.array(total_loss).mean()
    metric_dict["outputs"] = outputs.flatten()
    metric_dict["labels"] = labels.flatten()

    return metric_dict


def train(opt):
    print("===" * 30)
    print(f"{opt.label_name} Train Start!")

    savepath = os.path.join("./model", f"{opt.timeflag}", f"{opt.label_name}")
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    model_name = os.path.join(savepath, f"{opt.label_name}.pth")  # model
    params_name = os.path.join(savepath, "Train_params.pth")  # params
    writer = SummaryWriter(os.path.join(savepath, "runs"))  # tensorboard log
    log_file = open(os.path.join(savepath, "log.txt"), "w", encoding="utf-8")  # log train imformation

    # 设置 NumPy 的打印选项
    # np.set_printoptions(threshold=np.inf)
    np.set_printoptions(precision=4)

    torch.save(opt, params_name)  # 保存训练时设置的超参数
    print(str(opt))
    log_file.write(str(opt) + "\n")

    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # -------------------------Prepare Dataset-------------------------
    print("Initnializing Dataset...")

    # 加载数据集
    dataset = MyDataset(opt.datadir, opt.labeldir)

    # 划分训练集和验证集
    train_dataset, val_dataset = train_test_split(
        dataset, test_size=opt.val_rate, random_state=opt.seed
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, num_workers=6, shuffle=True, pin_memory=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=opt.batch_size * 2, num_workers=6, shuffle=False, pin_memory=True, prefetch_factor=2
    )

    # -------------------------Prepare Model-------------------------
    print("Initnializing Model...")

    # 模型初始化
    model = MyModel(in_c=8)

    # 优化模型
    # max-autotune最快但耗时，reduce-overhead适合加速小模型，需要额外存储空间，deffault适合大模型不需要额外存储空间
    # -----Cannot be used in Windows------
    # model_compile = torch.compile(model, mode="reduce-overhead")

    if torch.cuda.is_available():
        model.cuda(opt.gpu)
        # model_compile.cuda(opt.gpu)
    else:
        print("CUDA is not available. The model will run on CPU.")

    # -------------------------Prepare Optimizer and Lr_scheduler-------------------------
    print("Initnializing Optimizer...")
    if opt.optim == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
    elif opt.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optim == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=1)
    else:
        raise Exception(f"No {opt.optim} optimizer!")

    print("Initnializing Scheduler...")
    if opt.scheduler == "ReduceLROnPlateau":
        # mode: min/max, factor: 学习率缩放因子，patience: 没有进步的训练轮数，threshold: 测量新的最佳值时，阈值，min_lr: 学习率下限，verbose: 是否打印学习率变化
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            patience=4,
            factor=0.5,
            threshold=1e-4,
            min_lr=1e-9,
        )
    elif opt.scheduler == "CosineAnnealingLR":
        # T_max: 最大迭代次数，eta_min: 最小学习率, last_epoch: 最后一个epoch的index
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30, eta_min=0.00001, last_epoch=-1
        )
    elif opt.scheduler == "ExponentialLR":
        scheduler = lr_scheduler.ExponentialLR(
            optimizer, gamma=0.95, last_epoch=-1, verbose=False
        )
    else:
        raise Exception(f"No {opt.scheduler} scheduler!")

    print("===" * 30)

    if opt.warmup_lr_end != opt.lr:
        opt.warmup_lr_end = opt.lr
        opt.warmup_lr_init = opt.lr / 1000

    best_loss = 1000.0
    stopflag = 0
    # -------------------------Train-------------------------
    for i in range(1, opt.epoch):

        model.train()

        loss_epoch = []  # 每个epoch的loss
        outputs = []  # 每个epoch的输出
        labels = []  # 每个epoch的标签
        outputs_ = []
        labels_ = []

        # Model Warm Up
        if i <= opt.warmup_epoch:
            current_lr = opt.warmup_lr_init + (opt.warmup_lr_end - opt.warmup_lr_init) * (i / opt.warmup_epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        # Training
        for j, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {i} train"):
            data = data.cuda(opt.gpu)
            label = label.cuda(opt.gpu)

            output = model(data)
            # Loss
            loss = nn.MSELoss()(output, label)
            # loss = nn.SmoothL1Loss(beta=opt.beta)(output, label)  # SmoothL1Loss其相对于均方差(MSE)损失函数的优势在于对异常值(如过大或过小的离群点)的惩罚更小从而使模型更加健壮
            loss_epoch.append(loss.item())
            
            outputs_.append(output.detach()) 
            labels_.append(label.detach())  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (j+1) % 50 == 0:
                outputs.append(torch.cat(outputs_, dim=0).cpu())
                labels.append(torch.cat(labels_, dim=0).cpu())
                del outputs_, labels_
                torch.cuda.empty_cache()
                outputs_ = []
                labels_ = []

        # -------------------------Metric-------------------------
        if outputs_:
            outputs.append(torch.cat(outputs_, dim=0).cpu())
            labels.append(torch.cat(labels_, dim=0).cpu())
            del outputs_, labels_
        
        outputs = torch.cat(outputs, dim=0).numpy().flatten()
        labels = torch.cat(labels, dim=0).numpy().flatten()

        # 检查并处理NaN值
        if np.isnan(outputs).any() or np.isnan(labels).any():
            print("警告：发现NaN值，正在尝试处理...")
            outputs = np.nan_to_num(outputs)
            labels = np.nan_to_num(labels)
            
        # 评价指标
        metric_dict = eval_metric(outputs, labels)

        loss_train_epoch = np.array(loss_epoch).mean()
        lr = optimizer.param_groups[0]["lr"]  # get learning rate
        
        if i > opt.warmup_epoch:
            if opt.scheduler == "ReduceLROnPlateau":
                scheduler.step(loss_train_epoch)
            else:
                scheduler.step()

        # -------------------------Validate-------------------------
        val_metric = val_test(model, val_loader, opt)
        # torch.cuda.empty_cache()

        out_str = (
            f"{i:3} Epoch LR:{lr}\n"
            f"Train Loss:{loss_train_epoch:>7.4f} R2:{metric_dict['r2']:>7.4f} R:{metric_dict['r']:>7.4f} "
            f"MAE:{metric_dict['mae']:>7.4f} RMSE:{metric_dict['rmse']:>7.4f}\n"
            f"Valid Loss:{val_metric['loss']:>7.4f} R2:{val_metric['r2']:>7.4f} "
            f"R:{val_metric['r']:>7.4f} "
            f"MAE:{val_metric['mae']:>7.4f} "
            f"RMSE:{val_metric['rmse']:>7.4f} "
        )
        print(out_str)
        log_file.write(out_str + f"\n")

        # tensorboardx
        writer.add_scalar("Train/Loss", loss_train_epoch, i)
        writer.add_scalar("Train/R2", metric_dict["r2"], i)
        writer.add_scalar("Train/r", metric_dict["r"], i)
        writer.add_scalar("Train/MAE", metric_dict["mae"], i)
        writer.add_scalar("Train/RMSE", metric_dict["rmse"], i)
        writer.add_scalar("Train/LR", lr, i)
        writer.add_scalar("Val/Loss", val_metric['loss'], i)
        writer.add_scalar("Val/R2", val_metric['r2'], i)
        writer.add_scalar("Val/r", val_metric['r'], i)
        writer.add_scalar("Val/MAE", val_metric['mae'], i)
        writer.add_scalar("Val/RMSE", val_metric['rmse'], i)

        # 改进时保存
        if best_loss > loss_train_epoch:
            best_loss = loss_train_epoch
            torch.save(model.state_dict(), model_name)

            stopflag = 0
            print(f"----epoch {i} Model Saved! ------\n")
            log_file.write(f"epoch {i} Model Saved! \n")

            # # 绘制散点图
            # print("Draw Points...")
            # t1 = time.time()
            # minv, maxv = draw_points(outputs, labels, savepath, "Train")
            # draw_points(val_metric["outputs"], val_metric["labels"], savepath, "Validation", maxv=maxv, minv=minv)
            # t2 = time.time()
            # print(f"Draw Points Time: {t2-t1:.2f}s")
        
        del outputs, labels
        
        # stop early        
        stopflag += 1
        if stopflag > opt.early_stop:
            print(f"Early Stop at Epoch {i}")
            break
    
    # load best model
    model.load_state_dict(torch.load(model_name))
    model.eval()
    train_metric = val_test(model, train_loader, opt)
    val_metric = val_test(model, val_loader, opt)
    
    # write data
    with open(os.path.join(savepath, "data.txt"), "w") as data_file:
                data_file.write(f"Train:\n")
                for k in range(len(train_metric["outputs"])):
                    data_file.write(f"{train_metric['outputs'][k]:.4f},{train_metric['labels'][k]:.4f} ")
                    if k % 10 == 0 and k != 0:
                        data_file.write("\n")
                data_file.write("\n")
                
                data_file.write(f"\nVal:\n")
                for k in range(len(val_metric["outputs"])):
                    data_file.write(f"{val_metric['outputs'][k]:.4f},{val_metric['labels'][k]:.4f} ")
                    if k % 10 == 0 and k != 0:
                        data_file.write("\n")
                data_file.write("\n")
    
    log_file.write(out_str + "\n")

    print(f"{opt.label_name} Train Finish!")
    log_file.close()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    opt_ = parse_args()

    timeflag_ = datetime.now().strftime("%Y%m%d-%H%M%S")
    opt_.timeflag = timeflag_
    
    opt_.datadir = r"D:\SoilErosion\PPS_Downscaling\500m\Environment\npy_patches"
    labeldir = r"D:\SoilErosion\PPS_Downscaling\500m\Soil\npy_patches"

    opt_.epoch = 500
    opt_.batch_size = 8
    opt_.lr = 0.001
    opt_.early_stop = 15
    opt_.optim = "AdamW"
    opt_.val_rate = 0.3
    opt_.warmup_epoch = 5

    labels = ["Fenli", "Nianli", "Shali", "SOC"]

    for label in labels:
        opt_.label_name = label

        opt_.labeldir = os.path.join(labeldir, label)
        train(opt_)

    print("All Geochemistry Train Finish!")
