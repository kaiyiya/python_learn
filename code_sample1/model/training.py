from __future__ import division  # 兼容Python2/3的除法行为，确保1/2得到浮点数，例如1/2=0.5而不是整数0
import torch  # 导入PyTorch库，用于张量计算和深度学习
import time  # 导入time模块，用来计时每个epoch或step的耗时
import numpy as np  # 导入NumPy，用于数值计算，比如求均值
import os  # 导入os模块，负责文件夹创建等操作
import matplotlib.pyplot as plt  # 导入Matplotlib的绘图工具，用于保存预测与真值的可视化图像


class Trainer(object):  # 定义Trainer类，用来封装训练全过程
    def __init__(self, train_loader, model, opt, device, val_loader=None):  # 初始化函数，接收数据加载器、模型、配置等
        self.args = opt  # 保存额外配置参数，例如学习率、路径设置
        self.train_loader = train_loader  # 保存训练集的DataLoader，便于后续循环读取数据
        self.val_loader = val_loader  # 保存验证集的DataLoader，如果传入则用于训练后验证
        self.model = model  # 保存需要训练的模型对象
        self.device = device  # 保存训练所使用的设备（CPU或GPU）
        import torch.nn.functional as F  # 在构造函数内部导入F，避免全局污染，便于调用函数式API
        self.F = F  # 保存torch.nn.functional引用，后面用于计算损失
        self.criterion = None  # 预留损失函数位置，这里通过F临时构造，所以先置None示意
        self.optimizer = torch.optim.Adam(lr=0.0003, params=model.parameters())  # 使用Adam优化器，学习率0.0003，传入模型参数
        self.epochs = 200  # 设定训练总轮数为200，相当于循环200次训练集
        self.model.to(device)  # 将模型移动到指定的device，比如GPU，加速训练
        self.pos_weight = torch.tensor(16.0, device=self.device)  # 定义正样本权重pos_weight=16，在前景稀少时加大正例损失权
        self.pos_weight = torch.clamp(self.pos_weight, max=50.0)  # 对pos_weight做上限限制，避免权重过大导致训练不稳定

    def calculate_iou(self, pred, target, threshold=0.5):  # 定义IoU计算函数，用于评估分割结果
        """计算IoU (Intersection over Union)，示例：预测图与真值图重叠区域面积/并集面积"""  # 文档字符串说明含义，并给出直观示例
        pred_binary = (pred > threshold).float()  # 将预测概率图按阈值二值化，例如阈值0.5时概率>0.5记为1
        target_binary = (target > threshold).float()  # 将真值图同样按阈值二值化，保持比较方式一致

        intersection = (pred_binary * target_binary).sum()  # 交集部分是预测与真值同时为1的像素总和，可理解为重叠区域像素数
        union = pred_binary.sum() + target_binary.sum() - intersection  # 并集=预测区域+真值区域-交集，避免重复统计

        if union == 0:  # 如果预测和真值全是背景（并集=0），避免除0
            return torch.tensor(1.0).to(pred.device)  # 返回1.0表示完美重合，例如黑色背景与黑色背景完全一致
        return intersection / union  # 正常情况下返回IoU值，越接近1说明预测越准确

    def calculate_dice(self, pred, target, threshold=0.5, smooth=1e-6):  # 定义Dice系数计算函数
        """计算Dice系数，示例：Dice=2*交集/(预测面积+真值面积)，常用于前景稀少的医学影像"""  # 解释Dice含义和应用
        pred_binary = (pred > threshold).float()  # 阈值化预测结果
        target_binary = (target > threshold).float()  # 阈值化真值

        intersection = (pred_binary * target_binary).sum()  # 计算交集像素数
        dice = (2.0 * intersection + smooth) / (pred_binary.sum() + target_binary.sum() + smooth)  # 按公式计算Dice，并加平滑项防止除0
        return dice  # 返回Dice系数，值越接近1说明越重合

    def dice_loss(self, logits, target, smooth=1e-6):  # 定义Dice损失，用于训练过程
        """Dice Loss 基于概率（对logits做sigmoid），示例：若模型完全重合则Dice Loss≈0"""  # 说明Dice损失的直观意义
        probs = torch.sigmoid(logits)  # 将logits转换为概率，范围0~1，方便与真值比较
        intersection = (probs * target).sum()  # 计算概率与真值的交集，代表重叠程度
        union = probs.sum() + target.sum()  # 计算概率总和与真值总和，对应预测面积与真实面积
        dice = (2.0 * intersection + smooth) / (union + smooth)  # 用概率版Dice公式并加平滑项，避免空样本出错
        return 1.0 - dice  # Dice Loss=1-Dice，越小越好，完全重合时Loss接近0

    def calculate_accuracy(self, pred, target, threshold=0.5):  # 定义二分类准确率计算函数
        """计算准确率，示例：预测与真值完全一致时准确率=1.0"""  # 简要说明准确率
        pred_binary = (pred > threshold).float()  # 按阈值将预测转成0/1
        target_binary = (target > threshold).float()  # 将真值也转成0/1

        correct = (pred_binary == target_binary).float().sum()  # 统计预测与真值相同的像素数量
        total = pred_binary.numel()  # 总像素数量，即图像尺寸
        return correct / total  # 准确率=正确像素/总像素，例如正确100个、总数120个时准确率≈0.833

    def calculate_mae(self, pred, target):  # 定义平均绝对误差MAE
        """计算平均绝对误差 (MAE)，示例：统计预测概率与真值的平均差距"""  # 说明MAE含义
        return torch.abs(pred - target).mean()  # 计算预测与真值的绝对差并求平均，差越小说明预测越接近

    def get_gradient_norm(self, model):  # 定义计算模型梯度范数的函数
        """计算梯度范数，示例：用来观察梯度是否过大或接近0"""  # 说明梯度范数用途
        total_norm = 0.0  # 初始化梯度平方和
        for p in model.parameters():  # 遍历模型所有参数
            if p.grad is not None:  # 只有在梯度存在时才计算，避免None出错
                param_norm = p.grad.data.norm(2)  # 计算该参数的L2范数，即sqrt(sum(grad^2))
                total_norm += param_norm.item() ** 2  # 将范数平方累加，相当于累积每个参数的梯度能量
        total_norm = total_norm ** (1. / 2)  # 开平方得到整体梯度L2范数
        return total_norm  # 返回梯度大小，可用于监控训练是否稳定

    def train_model(self):  # 定义主训练函数
        print(f'开始训练，共 {self.epochs} 个epoch')  # 打印训练开始信息，提示总轮数，例如200轮
        print('=' * 80)  # 打印分割线，便于日志阅读

        for epoch in range(self.epochs):  # 遍历每个epoch，range(200)即0~199共200轮
            epoch_start_time = time.time()  # 记录本轮开始时间，用于统计耗时
            losses = []  # 初始化本轮所有batch的loss列表，用于后续求平均
            ious = []  # 存储IoU指标
            dices = []  # 存储Dice指标
            accuracies = []  # 存储准确率
            maes = []  # 存储MAE指标
            gradient_norms = []  # 存储每个step的梯度范数，观察梯度变化

            self.model.train()  # 设置模型为训练模式，启用dropout、BN统计等

            scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))  # 创建GradScaler，在GPU上启用混合精度自动缩放
            use_bf16 = (self.device.type == 'cuda') and hasattr(torch.cuda,
                                                                'is_bf16_supported') and torch.cuda.is_bf16_supported()  # 检查是否支持bfloat16，提高兼容性
            for i, (img, mask) in enumerate(self.train_loader):  # 遍历训练集的每个batch，i是批次索引
                step_start_time = time.time()  # 记录当前batch开始时间
                img, mask = img.to(self.device), mask.float().to(self.device)  # 将图像和掩码移动到设备上，并确保掩码为浮点数

                self.optimizer.zero_grad(set_to_none=True)  # 梯度清零，set_to_none=True节省内存并加速
                autocast_kwargs = {}  # 初始化autocast参数字典，默认空
                if use_bf16:  # 如果GPU支持bfloat16
                    autocast_kwargs = dict(dtype=torch.bfloat16)  # 设置autocast使用bfloat16精度，示例：RTX3090可支持
                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda'),
                                             **autocast_kwargs):  # 启用自动混合精度，降低计算耗时
                    output = self.model(img)  # logits  # 前向传播得到logits输出，未经过sigmoid
                    bce = self.F.binary_cross_entropy_with_logits(output, mask,
                                                                  pos_weight=self.pos_weight)  # 计算加权BCE损失，示例：正类越少权重越大
                    dice = self.dice_loss(output, mask)  # 计算Dice损失，示例：鼓励预测区域与真值重合
                    loss = bce + 1.0 * dice  # 总损失=BCE+Dice，系数1表示同等重要
                if not torch.isfinite(loss):  # 若损失出现NaN或Inf
                    print('  ⚠️  非有限loss，跳过该batch')  # 打印警告提醒，例如学习率太大或数值溢出
                    continue  # 跳过当前batch，避免反向传播出错
                scaler.scale(loss).backward()  # 使用scaler缩放loss再反向传播，保持混合精度稳定

                grad_norm = self.get_gradient_norm(self.model)  # 在优化前计算梯度范数，了解梯度大小
                gradient_norms.append(grad_norm)  # 将梯度范数记录下来，方便后续统计平均值

                scaler.unscale_(self.optimizer)  # 取消缩放，使梯度恢复原尺度，便于梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 对梯度做裁剪，防止梯度爆炸，例如限制最大范数为1
                scaler.step(self.optimizer)  # 使用缩放过的梯度更新参数d
                scaler.update()  # 更新scaler的缩放因子，自适应下一次迭代

                with torch.no_grad():  # 下面是评估指标与监控信息，不需要梯度
                    mask_sum = mask.sum().item()  # 计算掩码中正样本总和，示例：判断前景像素量
                    logits_sum = output.sum().item()  # 计算logits总和，可观察输出偏置
                    probs = torch.sigmoid(output)  # logits转为概率，用于指标计算
                    output_sum = probs.sum().item()  # 概率总和，查看模型输出整体偏向
                    mask_nonzero = (mask > 0.01).sum().item()  # 统计掩码中非零像素数量，判断前景稀疏程度

                    thr = 0.2  # 将评估阈值设为0.2，示例：在前景极少时提高召回率
                    iou = self.calculate_iou(probs, mask, threshold=thr).item()  # 计算IoU并转为Python数值
                    dice = self.calculate_dice(probs, mask, threshold=thr).item()  # 计算Dice指标
                    accuracy = self.calculate_accuracy(probs, mask, threshold=thr).item()  # 计算准确率
                    mae = self.calculate_mae(probs, mask).item()  # 计算MAE指标，用于衡量概率误差

                losses.append(loss.item())  # 记录当前batch的损失值
                ious.append(iou)  # 记录IoU
                dices.append(dice)  # 记录Dice
                accuracies.append(accuracy)  # 记录准确率
                maes.append(mae)  # 记录MAE

                step_time = time.time() - step_start_time  # 计算当前batch耗时
                samples_per_sec = img.size(
                    0) / step_time if step_time > 0 else 0  # 计算吞吐率，例如batch=8耗时0.4s则20 samples/sec

                current_lr = self.optimizer.param_groups[0]['lr']  # 读取优化器当前学习率，方便打印观察

                if i % 10 == 0:  # 每10个batch打印一次日志，避免输出过多
                    print(f'\n[Epoch {epoch}/{self.epochs}] [Step {i}/{len(self.train_loader)}]')  # 打印基本进度
                    print(f'  Loss: {loss.item():.6f}')  # 打印损失值，示例：0.123456
                    print(
                        f'  IoU: {iou:.4f} | Dice: {dice:.4f} | Acc: {accuracy:.4f} | MAE: {mae:.6f}')  # 打印各类指标，帮助判断模型表现
                    print(f'  LR: {current_lr:.6f} | Grad Norm: {grad_norm:.6f}')  # 打印学习率与梯度范数，用于调参
                    print(
                        f'  Speed: {samples_per_sec:.2f} samples/sec | Step Time: {step_time:.3f}s')  # 打印速度信息，示例：16 samples/sec
                    print(f'  Image range: [{img.min():.6f}, {img.max():.6f}]')  # 打印输入张量范围，检查是否归一化
                    print(
                        f'  Logits range: [{output.min():.6f}, {output.max():.6f}] | Probs range: [{probs.min():.6f}, {probs.max():.6f}] | Prob sum: {output_sum:.2f}')  # 打印输出范围和概率总和，查看模型输出是否正常
                    print(
                        f'  Mask range: [{mask.min():.6f}, {mask.max():.6f}] | Mask sum: {mask_sum:.2f} | Non-zero pixels: {mask_nonzero}')  # 打印掩码统计，判断数据质量

                    if grad_norm < 1e-6:  # 梯度几乎为0时
                        print(f'  ⚠️  警告: 梯度范数接近0，模型可能已停止更新！')  # 提醒可能陷入梯度消失
                    if probs.max() < 1e-3:  # 如果输出概率极低
                        print(f'  ⚠️  警告: 概率输出几乎全为0，可能模型塌陷！')  # 提醒模型可能只输出背景
                    if mask_nonzero < mask.numel() * 0.01:  # 前景像素低于1%
                        print(f'  ⚠️  警告: 掩码几乎全为背景（非零像素<1%），数据可能有问题！')  # 提示数据也许需要清洗或检查

            epoch_time = time.time() - epoch_start_time  # 计算本轮训练耗时
            avg_loss = np.mean(losses)  # 统计所有batch平均损失，例如列表[0.2,0.18]取均值
            avg_iou = np.mean(ious)  # 统计平均IoU
            avg_dice = np.mean(dices)  # 统计平均Dice
            avg_accuracy = np.mean(accuracies)  # 统计平均准确率
            avg_mae = np.mean(maes)  # 统计平均MAE
            avg_grad_norm = np.mean(gradient_norms)  # 统计平均梯度范数，用以观测趋势
            total_samples = len(self.train_loader.dataset)  # 训练集中样本总数，示例：1000张图
            epoch_speed = total_samples / epoch_time if epoch_time > 0 else 0  # 计算每秒处理样本数，衡量整体训练速度

            print(f'\n{"=" * 80}')  # 打印分隔线，区分不同epoch
            print(f'Epoch {epoch}/{self.epochs} 总结:')  # 输出当前epoch编号
            print(f'  平均 Loss: {avg_loss:.6f}')  # 打印平均损失
            print(f'  平均 IoU: {avg_iou:.4f}')  # 打印平均IoU
            print(f'  平均 Dice: {avg_dice:.4f}')  # 打印平均Dice
            print(f'  平均准确率: {avg_accuracy:.4f}')  # 打印平均准确率
            print(f'  平均 MAE: {avg_mae:.6f}')  # 打印平均MAE
            print(f'  平均梯度范数: {avg_grad_norm:.4f}')  # 打印平均梯度范数
            print(f'  当前学习率: {self.optimizer.param_groups[0]["lr"]:.6f}')  # 打印当前学习率，确认是否有调度
            print(f'  训练时间: {epoch_time:.2f}s | 训练速度: {epoch_speed:.2f} samples/sec')  # 打印每轮耗时和速度
            print(f'{"=" * 80}')  # 再次打印分隔线，保持日志整洁

            try:  # 使用try避免可视化过程中出现错误导致训练中断
                os.makedirs('PredvsGT', exist_ok=True)  # 创建保存图片的文件夹PredvsGT，若已存在则忽略
                with torch.no_grad():  # 可视化阶段不需要梯度
                    for vis_img, vis_mask in self.train_loader:  # 从训练集中取一批次用于可视化
                        vis_img = vis_img.to(self.device)  # 将可视化图像搬到设备
                        vis_mask = vis_mask.to(self.device)  # 同样处理真值
                        vis_logits = self.model(vis_img)  # 前向计算得到logits
                        vis_probs = torch.sigmoid(vis_logits)  # 转为概率图
                        thr = 0.2  # 可视化使用的阈值，保持与上文一致
                        vis_pred = (vis_probs > thr).float()  # 阈值化得到二值预测图

                        img_np = vis_img[0, 0].detach().cpu().numpy()  # 取batch第一个样本第一通道，并转为numpy，示例：单通道灰度图
                        prob_np = vis_probs[0, 0].detach().cpu().numpy()  # 同样取概率图用于绘制
                        pred_np_t02 = vis_pred[0, 0].detach().cpu().numpy()  # 阈值0.2下的预测结果
                        pred_np_t05 = (vis_probs[0, 0] > 0.5).float().cpu().numpy()  # 阈值0.5下的预测结果，方便对比
                        mask_np = vis_mask[0, 0].detach().cpu().numpy()  # 获取真值掩码

                        fig, axes = plt.subplots(1, 4, figsize=(18, 4))  # 创建1行4列子图，展示不同图像
                        axes[0].imshow(img_np, cmap='gray');
                        axes[0].set_title('Input');
                        axes[0].axis('off')  # 显示原始输入图像，关闭坐标轴
                        im1 = axes[1].imshow(prob_np, cmap='viridis', vmin=0.2, vmax=0.8)  # 显示概率图，并在0.2~0.8之间着色
                        axes[1].set_title('Prob (0.2~0.8)');
                        axes[1].axis('off');
                        plt.colorbar(im1, ax=axes[1])  # 添加颜色条用于读取概率值
                        axes[2].imshow(pred_np_t02, cmap='gray');
                        axes[2].set_title('Pred th=0.2');
                        axes[2].axis('off')  # 显示阈值0.2的预测结果
                        axes[3].imshow(mask_np, cmap='gray');
                        axes[3].set_title('GT');
                        axes[3].axis('off')  # 显示真值掩码
                        plt.tight_layout()  # 调整子图间距，避免重叠
                        plt.savefig(os.path.join('PredvsGT', f'epoch_{epoch:03d}.png'), dpi=150,
                                    bbox_inches='tight')  # 保存图片，文件名包含epoch编号
                        plt.close()  # 关闭图像释放内存

                        fig2, axes2 = plt.subplots(1, 4, figsize=(22, 4))  # 创建第二组4图，用于差异可视化
                        axes2[0].imshow(img_np, cmap='gray')  # 显示原始图像作为底图
                        c1 = axes2[0].contour(mask_np, levels=[0.5], colors='lime', linewidths=1)  # 绘制真值轮廓，绿色示例：真实病灶边界
                        c2 = axes2[0].contour(pred_np_t02, levels=[0.5], colors='red',
                                              linewidths=1)  # 绘制预测轮廓，红色示例：模型预测边界
                        axes2[0].set_title('Overlay: GT(green) & Pred(red)')  # 标题说明颜色含义
                        axes2[0].axis('off')  # 关闭坐标轴

                        axes2[1].imshow(pred_np_t02, cmap='gray');
                        axes2[1].set_title('Pred th=0.2');
                        axes2[1].axis('off')  # 显示阈值0.2预测
                        axes2[2].imshow(pred_np_t05, cmap='gray');
                        axes2[2].set_title('Pred th=0.5');
                        axes2[2].axis('off')  # 显示阈值0.5预测，用于比较阈值影响

                        fp = (pred_np_t02 == 1) & (mask_np == 0)  # 找出假阳性区域，例如预测病灶但实际背景
                        fn = (pred_np_t02 == 0) & (mask_np == 1)  # 找出假阴性区域，例如漏检病灶
                        tp = (pred_np_t02 == 1) & (mask_np == 1)  # 找出真阳性区域，例如正确覆盖病灶
                        h, w = mask_np.shape  # 获取图像高度和宽度，用于创建彩色差异图
                        diff = np.zeros((h, w, 3), dtype=np.float32)  # 初始化RGB差异图，默认全黑
                        diff[fp] = [1.0, 0.0, 0.0]  # 将假阳性区域标红
                        diff[fn] = [0.0, 0.0, 1.0]  # 将假阴性区域标蓝
                        diff[tp] = [0.0, 1.0, 0.0] * 0.3  # 将真阳性区域涂成淡绿色，让用户直观看到正确区域
                        axes2[3].imshow(diff)  # 显示差异图
                        axes2[3].set_title('Diff: FP(red) FN(blue) TP(green)')  # 给出图例说明
                        axes2[3].axis('off')  # 关闭坐标轴
                        plt.tight_layout()  # 调整布局
                        plt.savefig(os.path.join('PredvsGT', f'epoch_{epoch:03d}_diff.png'), dpi=150,
                                    bbox_inches='tight')  # 保存差异图
                        plt.close()  # 关闭图像
                        break  # 只可视化第一个batch，防止耗时过多
            except Exception:  # 捕获可视化过程的所有异常，避免影响训练流程
                pass  # 如果失败则忽略，不影响训练主流程

            if self.val_loader is not None:  # 如果存在验证集，则执行验证
                self.model.eval()  # 切换到评估模式，关闭Dropout等
                val_losses = []  # 初始化验证集损失列表
                val_ious = []  # 初始化验证集IoU列表
                val_dices = []  # 初始化验证集Dice列表
                val_accuracies = []  # 初始化验证集准确率列表
                val_maes = []  # 初始化验证集MAE列表
                with torch.no_grad():  # 验证阶段不需要梯度
                    for img, mask in self.val_loader:  # 遍历验证集
                        img, mask = img.to(self.device), mask.float().to(self.device)  # 将验证数据拷贝到设备
                        with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda'),
                                                     **autocast_kwargs):  # 继续使用混合精度，保持一致
                            output = self.model(img)  # logits  # 前向传播得到logits
                            bce_v = self.F.binary_cross_entropy_with_logits(output, mask,
                                                                            pos_weight=self.pos_weight)  # 验证BCE损失
                            dice_v = self.dice_loss(output, mask)  # 验证Dice损失
                            vloss = bce_v + 1.0 * dice_v  # 验证总损失
                        val_losses.append(vloss.item())  # 保存损失值
                        probs = torch.sigmoid(output)  # logits转概率
                        thr = 0.2  # 验证阶段同样使用阈值0.2
                        val_ious.append(self.calculate_iou(probs, mask, threshold=thr).item())  # 记录IoU
                        val_dices.append(self.calculate_dice(probs, mask, threshold=thr).item())  # 记录Dice
                        val_accuracies.append(self.calculate_accuracy(probs, mask, threshold=thr).item())  # 记录准确率
                        val_maes.append(self.calculate_mae(probs, mask).item())  # 记录MAE

                print(f'验证集:')  # 打印验证结果开头
                print(f'  平均 Val Loss: {np.mean(val_losses):.6f}')  # 打印验证集平均损失
                print(f'  平均 Val IoU: {np.mean(val_ious):.4f}')  # 打印验证集平均IoU
                print(f'  平均 Val Dice: {np.mean(val_dices):.4f}')  # 打印验证集平均Dice
                print(f'  平均 Val Acc: {np.mean(val_accuracies):.4f}')  # 打印验证集平均准确率
                print(f'  平均 Val MAE: {np.mean(val_maes):.6f}')  # 打印验证集平均MAE
                print(f'{"=" * 80}\n')  # 打印分隔线并换行
            else:  # 如果没有验证集
                print()  # 打印空行，以保持输出格式一致

            if epoch % 10 == 0:  # 每10个epoch保存一次模型
                torch.save(self.model.state_dict(), f'unet-{epoch}.pth')  # 保存当前模型参数到文件，例如unet-010.pth
                print(f'✓ 模型已保存: unet-{epoch}.pth\n')  # 提示保存成功，方便回滚特定阶段

        torch.save(self.model.state_dict(), f'Unet-epochs{self.epochs}.pth')  # 所有训练结束后再保存一次最终权重
        print(f'\n训练完成！最终模型已保存: Unet-epochs{self.epochs}.pth')  # 打印训练完成提示，告知模型存放位置
