import wandb

# 初始化一个新的项目
wandb.init(project='test_project')

# 记录超参数和结果
config = wandb.config
config.learning_rate = 0.01
config.epochs = 10

for epoch in range(config.epochs):
    # 模拟训练过程
    loss = 0.01 * epoch  # 这是一个示例，请替换为实际的损失值

    # 记录损失
    wandb.log({'epoch': epoch, 'loss': loss})

print("wandb logging test completed successfully")