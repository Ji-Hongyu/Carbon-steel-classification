import torchvision.models
from carbon_steel_classification.utils import MyDataset
from model import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
train_dataset = MyDataset('./data_divided_into6/train')
val_dataset = MyDataset('./data_divided_into6/val')

train_dataset_size = len(train_dataset)
val_dataset_size = len(val_dataset)

print(train_dataset_size)
print(val_dataset_size)

train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=True)

print('Done dataset loading')

# 创建网络模型
# vgg16 = VGG16()                                             自建模型
vgg16 = torchvision.models.vgg16()                          # torchvision中提供的vgg16模型，自行训练全部权重
# vgg16 = torchvision.models.vgg16(weights='DEFAULT')       # torchvision中提供的vgg16模型，采用预训练权重，仅最后全连接层权重自行训练

vgg16.classifier[6] = nn.Sequential(nn.Linear(4096, 6))     # 替换分类器最后一层全连接层，分类数由 1000 调整为 6
if torch.cuda.is_available():
    vgg16 = vgg16.cuda()

# 训练配置信息
epochs = 100
learning_rate = 1e-4
total_train_step = 0
total_test_step = 0

# 添加Tensorboard
writer = SummaryWriter("./logs")

# 优化器
optimizer = torch.optim.SGD(vgg16.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8, last_epoch=-1)  # 设置学习率衰减

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 开始训练
for epoch in range(epochs):
    print("------第 {} 轮训练开始------".format(epoch + 1))
    # 训练步骤开始
    vgg16.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(torch.float32)
        targets = targets.to(torch.long)
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        imgs = torch.reshape(imgs, (20, 3, 224, 224))

        outputs = vgg16(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if total_train_step % 10 == 0:
            print("| 训练次数： {} | Loss: {} |".format(total_train_step, loss.item()))
        if total_train_step % 5 == 0:
            writer.add_scalar("Train_Loss", loss.item(), total_train_step)
        total_train_step += 1

    # 测试步骤开始
    vgg16.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in val_dataloader:
            imgs, targets = data
            imgs = imgs.to(torch.float32)
            targets = targets.to(torch.long)
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            imgs = torch.reshape(imgs, (20, 3, 224, 224))

            outputs = vgg16(imgs)
            loss = loss_fn(outputs, targets)

            total_test_loss += loss.item()
            total_accuracy += (outputs.argmax(1) == targets).sum()
        print('*************************************')
        print("| 第 {} 轮训练 | 整体测试集上的Loss: {}  |".format(epoch + 1, total_test_loss))
        print("| 第 {} 轮训练 | 整体测试集上的正确率: {} |".format(epoch + 1, total_accuracy / val_dataset_size))
        print('*************************************')
        total_test_step += 1
        writer.add_scalar("Test_Loss", total_test_loss, total_test_step)
        writer.add_scalar("Test_Accuracy", total_accuracy / val_dataset_size, total_test_step)

    if (epoch + 1) % 10 == 0:
        torch.save(vgg16.state_dict(), "./saved/6_Nopre/6_Nopre_vgg16_state_{}.pth".format(int((epoch + 1) / 10)))
        print("----------模型 {} 已保存----------".format(int((epoch + 1) / 10)))

    scheduler.step()

writer.close()
