# AGENTS.md

本文件记录 echemistpy-cli 的协作和代码修改规则。优先遵守用户当前请求；若请求与本文件冲突，以当前请求为准。

## 1. 代码边界

- `data` 只放数据模型、schema、标准化、存储和通用数据工具。
- `io` 只负责 reader/writer 门面、加载入口、格式识别和插件注册；具体写出细节委托给 `data.storage`。
- `analysis` 只消费 `DataBundle`，返回 `AnalysisBundle`；分析模块不直接读取文件，也不承担绘图入口。
- `cli` 是薄壳，只做参数解析、调用 public API 和输出用户可读结果。

## 2. 数据接口

- 统一使用 `Metadata`、`DataBundle`、`AnalysisBundle`。
- 不恢复 `RawData`、`RawDataInfo`、`ResultsData` 等旧接口，也不写兼容层。
- echem 是主要开发主线；XAS、TXM、XRD 可作为可选能力逐步补齐，但不能破坏 echem 测试。

## 3. 依赖规则

- 核心代码不使用 `traitlets`。
- 默认依赖只保留数据层、CLI 和通用科学计算运行依赖。
- echem、XAS、TXM、XRD 的 reader 或分析专属依赖放入对应 extras；默认依赖中已有的包不要在 extras 中重复声明。
- 开发工具放入 `dev` 或 `test` extras，不放入默认依赖。

## 4. 语言规则

- 新增或修改的注释、docstring、CLI help、错误信息和 logging 文本使用中文。
- 保留代码标识符、标准列名、文件格式名和第三方库名称的英文原文。
- 日志使用结构化参数写法，例如 `logger.warning("读取 %s 失败: %s", path, exc)`。

## 5. 修改方式

- 改动应直接服务当前任务，避免顺手重构无关模块。
- 优先沿用现有模块结构和命名规则。
- 删除依赖或旧接口时，同步更新文档、pyproject 和测试。
- 不回滚用户已有改动；遇到无关脏文件时保持原样。

## 6. 验证

- echem 相关改动至少运行项目测试或与改动范围匹配的子集。
- 收尾前优先运行 `uv run ruff check` 检查代码格式和 lint 问题，运行 `uv run ty check` 检查类型问题；若只能运行子集或检查失败，需要说明范围和剩余问题。
- 依赖或 public API 改动后，扫描残留导入和旧接口名称。
- 如果完整 lint 因历史债务失败，应明确说明剩余失败位置，不把它混入当前改动。
