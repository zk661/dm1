import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# --- 步骤 1: 复用必要的代码 ---

# a) 复用模型定义：这个结构必须与训练时保存的模型结构完全一致
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc(x)
        return x


# b) 复用数据加载部分：我们需要测试数据集来评估模型
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True  # 如果数据不存在，会自动下载
)

test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# --- 步骤 2: 加载模型并进行评估 ---

def evaluate_saved_model():
    # 定义模型权重文件的路径
    model_path = "mnist_cnn.pth"

    # 1. 创建一个模型实例
    model = SimpleCNN()

    # 2. 加载已保存的权重
    print(f"正在从 {model_path} 加载模型权重...")
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print(f"错误: 模型文件 {model_path} 未找到。")
        print("请先运行 main.py 来训练并保存模型。")
        return

    # 3. 将模型设置为评估模式
    model.eval()

    # 4. 在整个测试集上评估模型
    correct = 0
    total = 0

    # 在评估时，我们不需要计算梯度，可以节省计算资源
    with torch.no_grad():
        for data, target in test_loader:
            # 前向传播
            output = model(data)
            # 获取预测结果 (概率最高的那个类别的索引)
            _, predicted = torch.max(output.data, 1)
            # 累加样本总数
            total += target.size(0)
            # 累加预测正确的样本数
            correct += (predicted == target).sum().item()

    # 5. 计算并打印最终的准确率
    accuracy = 100 * correct / total
    print(f"\n模型在 {total} 个测试样本上的准确率: {accuracy:.2f}%")
    print(f"预测正确数: {correct}, 总数: {total}")


# --- 步骤 3: 执行评估 ---
if __name__ == '__main__':
    evaluate_saved_model()

