# LM-GO: Go语言文本生成模型推理系统

## 项目简介

LM-GO 是一个使用 Go 语言从零实现的文本生成模型推理系统。该项目实现了完整的 Transformer 架构，支持加载和运行预训练的 Llama 格式模型，能够进行文本生成任务。参考：https://github.com/LearningInfiniTensor/learning-lm-rs

## 项目结构

```
learning-lm-go/
├── main.go              # 主程序入口
├── go.mod               # Go模块依赖
├── model/               # 模型相关代码
│   ├── model.go         # Llama模型实现
│   ├── params.go        # 模型参数加载
│   └── model_test.go    # 模型测试
├── tensor/              # 自研张量库
│   ├── tensor.go        # 张量基础实现
│   ├── operators.go     # 张量运算操作
│   └── tensor_test.go   # 张量测试
├── kvcache/             # KV缓存实现
│   └── kvcache.go       # 注意力机制缓存
├── models/              # 模型文件目录
│   └── story/           # Story生成模型
│       ├── config.json      # 模型配置
│       ├── model.safetensors # 模型权重
│       ├── tokenizer.json   # 分词器
│       └── tokenizer_config.json
└── build/               # 构建输出目录
```

## 核心功能

### 1. 自研张量库
- 支持多维张量操作和矩阵运算
- 实现泛型设计，支持多种数据类型
- 包含完整的单元测试覆盖

### 2. 模型加载与解析
- 实现 SafeTensors 格式解析器
- 支持从 HuggingFace 格式加载预训练模型
- 集成 tokenizers 库实现分词器功能

### 3. Transformer 架构实现
- **Self-Attention 机制**：多头注意力、RoPE 位置编码
- **FFN 层**：SwiGLU 激活函数、RMSNorm 归一化
- **Grouped Attention**：支持分组注意力机制

### 4. KV 缓存优化
- 实现高效的 Key-Value 缓存机制
- 支持增量推理，避免重复计算

### 5. 文本生成引擎
- 支持 temperature、top-p、top-k 等采样策略
- 完整的文本生成 pipeline

## 模型配置

当前项目包含一个 Story 生成模型，配置如下：
- **层数**：2 层 Transformer
- **注意力头**：8 个查询头，4 个键值头
- **隐藏层维度**：128
- **中间层维度**：384
- **词汇表大小**：2048
- **最大序列长度**：512

## 依赖项

- Go 1.24.4+
- github.com/daulet/tokenizers v1.20.2
- github.com/sirupsen/logrus v1.9.3
- github.com/antonfisher/nested-logrus-formatter v1.3.1

## 使用方法

1. **克隆项目**
```bash
git clone <repository-url>
cd learning-lm-go
```

2. **运行主程序**
```bash
go run main.go
```

程序将加载 `models/story` 目录下的模型，并根据输入提示生成文本。

## 测试

运行测试套件：
```bash
go test ./...
```

测试覆盖张量运算、模型加载、前向传播等核心功能。

## 项目特点

- **从零实现**：不依赖深度学习框架，完全自研
- **高性能**：优化的张量运算和内存管理
- **完整测试**：全面的单元测试覆盖
- **实际应用**：成功运行 Story 生成模型

## 技术栈

- **语言**：Go 1.24.4
- **架构**：Transformer (Llama 格式)
- **张量库**：自研实现
- **分词器**：HuggingFace tokenizers
- **日志**：logrus

## 开发状态

项目已完成核心功能实现，能够成功加载和运行预训练模型进行文本生成。代码结构清晰，包含完整的测试覆盖。

## 许可证

[待补充]

## 贡献

欢迎提交 Issue 和 Pull Request。 