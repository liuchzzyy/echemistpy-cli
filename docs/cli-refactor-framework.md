# echemistpy-cli CLI 重构设计框架

日期：2026-06-02

## 1. 结论

`echemistpy-cli` 当前更接近一个“可导入的科学数据处理原型库”，而不是一个成熟 CLI 库。它已经有有价值的基础能力：多格式 reader、`RawData/RawDataInfo` 容器、`xarray.Dataset/DataTree` 数据承载、标准化映射、分析 pipeline/registry 雏形。但当前不适合直接作为 CLI 发布，原因是：

- 没有 CLI 入口：`pyproject.toml` 未配置 `[project.scripts]`，源码中也没有 `cli.py`、`__main__.py`、Typer/Click/argparse 命令层。
- 分析层当前不可完整运行：`AnalysisPipeline()` 实例化失败，`create_default_registry()` 从子包根导入 analyzer，但子包没有 `__init__.py` 导出；`echem/analyzer.py` 也使用了错误的相对导入。
- 数据契约不一致：I/O 标准化会把 XAS 变量改成 `energy_eV/absorption_au`，但 XAS analyzer 要求 `energyc/absorption`。
- 插件系统可用但脆弱：默认插件能被扫描注册，但注册依赖 import side effect、类名推断和 `_get_file_extension()`，缺少显式插件 manifest/协议。
- reader 类职责过大：解析、清洗、标准化、元数据提取、目录聚合、DataTree 构造、仪器特殊逻辑混在一起，难以测试和组合成 CLI 工作流。
- 依赖过重：核心包依赖 `xraylarch`、`umap-learn`、`numba`、`scikit-image` 等重型分析库；CLI 用户只做 `load/list/export` 时不应安装所有分析依赖。
- 测试不可用：`pyproject.toml` 定义了 pytest，但当前 `.venv` 没有 `pytest`；仓库没有 `tests/`。

因此建议做“大重构”，但不是直接改代码。应先以本文档为蓝图，按阶段把项目改成“核心库 + CLI 外壳 + 插件扩展点”的结构。

## 2. 当前代码盘点

### 2.1 项目元数据

- 包发行名：`echemistpy-cli`
- Python 导入名：`echemistpy`
- 当前版本：`0.1.0`
- 构建后端：`setuptools`
- 源码布局：`src/echemistpy`
- 当前没有 `[project.scripts]`
- 当前没有 CLI 命令模块
- 当前没有测试目录

`pyproject.toml` 中 `dev` extra 引用了 `echemistpy-cli[interactive]` 和 `echemistpy-cli[docs]`，但这两个 optional extras 未定义，应在重构时修正。

### 2.2 当前模块

```text
src/echemistpy/
  __init__.py
  io/
    structures.py
    base_reader.py
    plugin_manager.py
    loaders.py
    standardizer.py
    column_mappings.py
    reader_utils.py
    saver.py
    plugins/
      Echem_BiologicMPTReader.py
      Echem_LanheXLSXReader.py
      XAS_CLAESS.py
      TXM_MISTRAL.py
      XRD_MSPD.py
  analysis/
    registry.py
    pipeline.py
    echem/analyzer.py
    stxm/analyzer.py
    xas/analyzer.py
    xas/processing.py
    xas/fitting.py
    xas/plotting.py
    xas/elements.py
scripts/
  workflow_operando.py
```

### 2.3 已验证行为

使用本地 `.venv/Scripts/python.exe` 和 `PYTHONPATH=src` 验证：

- `import echemistpy` 成功。
- `from echemistpy.io import list_supported_formats` 成功。
- 默认 I/O 插件可列出：`.mpt`, `.xlsx`, `.hdf5`, `.dat`, `.xye`。
- `from echemistpy.analysis.pipeline import AnalysisPipeline` 成功。
- `AnalysisPipeline()` 失败，错误为无法从 `echemistpy.analysis.echem` 导入 `GalvanostaticAnalyzer`。
- `from echemistpy.analysis.echem.analyzer import GalvanostaticAnalyzer` 失败，错误为 `No module named 'echemistpy.analysis.echem.registry'`。
- `.venv` 中没有 `pytest`。
- `ruff check src scripts` 当前报告 70 个问题。

## 3. 现有框架评估

### 3.1 可保留的部分

- `RawData`, `RawDataInfo`, `AnalysisData`, `AnalysisDataInfo` 的概念是合理的。
- `xarray.Dataset` 和 `xarray.DataTree` 适合承载多维科学数据和目录层级数据。
- I/O 插件思路正确：不同仪器 reader 解耦，比在 `load()` 中写硬编码分支更可扩展。
- `load(path, instrument=..., technique=...)` 作为 Python API 很实用。
- `save_data/save_info/save_combined` 可以成为 CLI `export` 的底层能力。
- XAS 的 `processing/fitting/plotting` 函数应保留为可复用算法模块。
- `AnalysisPipeline` 的方向正确：CLI 不应该直接操作复杂 analyzer，而应调用统一服务。

### 3.2 必须重构的部分

- CLI 层不存在，必须新增。
- analysis package 边界缺失，必须补齐 `__init__.py` 和导出策略。
- `TechniqueRegistry` 当前只按单一 technique 字符串匹配，和 reader 产生的 technique list 不完全适配。
- reader 插件注册缺少显式元信息，应从“类名推断”改为“插件类声明”。
- 数据标准名必须统一，不能 reader、standardizer、analyzer 各用一套变量名。
- 标准化不应默认破坏 analyzer 契约。例如 XAS loader 输出 `energyc/absorption`，standardizer 又改成 `energy_eV/absorption_au`，会导致 analyzer 失效。
- `scripts/workflow_operando.py` 仍引用旧包名 `claess2025`，不能作为当前项目示例。
- `traitlets` 用在普通数据容器和 reader 配置上增加复杂度，但收益有限；需要决定保留还是替换为 `dataclasses/pydantic`。
- `BaseReader._get_file_extension()` 反向查询全局 plugin manager，是不必要且脆弱的依赖方向。
- 错误处理粒度不稳定，很多 reader 捕获 `Exception` 后跳过文件，CLI 需要可配置的 strict/lenient 模式和明确退出码。

## 4. 重构目标

### 4.1 产品目标

把项目做成一个可长期扩展的 CLI 库：

```bash
echem formats
echem inspect data.mpt
echem load data.mpt --instrument biologic --out raw.nc
echem export data.mpt --format csv --out data.csv
echem analyze data.mpt --technique gcd --out analysis.nc
echem workflow operando-xas ./data --config workflow.yaml --out results/
```

CLI 应该是“薄壳”：参数解析、日志、错误输出、退出码、配置文件解析。核心业务逻辑必须在 Python API 层完成，保证未来可以被 notebook、脚本、GUI 或其他 agent 调用。

### 4.2 工程目标

- 可运行：基础 import、format listing、load、save、analyze 都可验证。
- 可测试：每个 reader 至少有 fixture 和 smoke test。
- 可扩展：新增 reader/analyzer 不需要改 CLI 主逻辑。
- 可裁剪依赖：最小安装只支持 CLI 基础和轻量 I/O；重型分析依赖放到 extras。
- 可追踪：CLI 输出结构化日志，失败时给出清晰原因。
- 可维护：reader/analyzer/service 分层，避免单类过长。

### 4.3 非目标

- 第一阶段不追求完整科学算法准确性重写。
- 第一阶段不做 GUI。
- 第一阶段不把所有 legacy workflow 迁入 CLI。
- 第一阶段不需要一次性支持所有仪器的高级分析。

## 5. 推荐目标架构

```text
src/echemistpy/
  cli/
    __init__.py
    app.py
    commands/
      formats.py
      inspect.py
      load.py
      export.py
      analyze.py
      workflow.py
    output.py
    errors.py
  core/
    models.py
    result.py
    exceptions.py
    logging.py
    paths.py
  io/
    api.py
    contracts.py
    registry.py
    standard_names.py
    standardize.py
    save.py
    readers/
      biologic_mpt.py
      lanhe_xlsx.py
      claess_dat.py
      mistral_hdf5.py
      mspd_xye.py
  analysis/
    api.py
    contracts.py
    registry.py
    echem/
      galvanostatic.py
    xas/
      analyzer.py
      processing.py
      fitting.py
      plotting.py
    stxm/
      analyzer.py
  workflows/
    operando_xas.py
  plugins/
    discovery.py
```

### 5.1 分层原则

- `cli`: 只做命令行交互，不做科学计算。
- `core`: 放通用数据模型、异常、结果对象、日志和路径工具。
- `io`: 只负责把文件变成标准 `RawDataBundle`，以及保存/导出。
- `analysis`: 只负责把标准数据变成分析结果。
- `workflows`: 组合多个 I/O 和 analysis service，适合复杂场景。
- `plugins`: 负责发现第三方 reader/analyzer。

## 6. 数据模型设计

### 6.1 推荐核心模型

当前 `RawData` 和 `RawDataInfo` 可以保留概念，但建议改成更明确的 bundle：

```python
@dataclass
class RawDataBundle:
    data: xr.Dataset | xr.DataTree
    info: RawDataInfo

@dataclass
class AnalysisBundle:
    data: xr.Dataset | xr.DataTree
    info: AnalysisDataInfo
```

原因：

- CLI 和 service 返回一个对象比返回 tuple 更稳定。
- 后续可以给 bundle 增加 `source_path`, `warnings`, `provenance`, `schema_version`。
- 旧 API 可保留兼容包装：`load()` 继续返回 `(RawData, RawDataInfo)`，新 API 返回 bundle。

### 6.2 标准变量命名

需要建立唯一标准命名表，并让 reader、standardizer、analyzer 全部依赖它。

建议第一版标准：

```text
通用：
  record
  time_s
  systime

电化学：
  cycle_number
  step_number
  ewe_v
  ece_v
  voltage_v
  current_ma
  capacity_mah
  specific_capacity_mah_g
  frequency_hz
  re_z_ohm
  neg_im_z_ohm

XAS：
  energy_ev
  absorption
  norm_absorption
  e0_ev

XRD：
  two_theta_deg
  intensity
  intensity_error
  d_spacing_angstrom

TXM/STXM：
  energy_ev
  x_um
  y_um
  transmission
  optical_density
```

重点：不要在标准名中使用 `/`, `%`, `-` 开头等 NetCDF/DataTree 不友好的字符。当前 `-im_z_ohm`、`2theta_degree`、`d-spacing_angstrom` 都应调整。

### 6.3 元数据设计

建议把元数据分为四类：

```text
identity:
  sample_name
  technique
  instrument

acquisition:
  start_time
  operator
  source_path
  file_count

sample:
  active_material_mass_g
  wavelength_angstrom

provenance:
  reader_name
  reader_version
  standard_schema_version
  warnings
  raw_metadata
```

`others` 可以保留，但应作为 `raw_metadata`，避免标准字段和动态字段混在一起。

## 7. 插件设计

### 7.1 Reader 协议

每个 reader 应显式声明能力，不依赖类名推断：

```python
class ReaderSpec(TypedDict):
    name: str
    extensions: tuple[str, ...]
    instruments: tuple[str, ...]
    techniques: tuple[str, ...]
    supports_directory: bool

class DataReader(Protocol):
    spec: ClassVar[ReaderSpec]

    def read(self, path: Path, options: ReaderOptions) -> RawDataBundle:
        ...
```

reader 注册时只读 `spec`：

```python
registry.register(BiologicMptReader)
registry.register(LanheXlsxReader)
```

### 7.2 Analyzer 协议

Analyzer 也应显式声明输入契约：

```python
class AnalyzerSpec(TypedDict):
    name: str
    techniques: tuple[str, ...]
    instruments: tuple[str, ...] | None
    required_variables: tuple[str, ...]
    output_schema: str

class Analyzer(Protocol):
    spec: ClassVar[AnalyzerSpec]

    def analyze(self, bundle: RawDataBundle, options: AnalysisOptions) -> AnalysisBundle:
        ...
```

这样 `analyze` 命令可以先做 dry-run 校验：

```bash
echem analyze data.mpt --technique gcd --check
```

### 7.3 第三方插件

未来可用 entry points：

```toml
[project.entry-points."echemistpy.readers"]
biologic_mpt = "echemistpy.io.readers.biologic_mpt:BiologicMptReader"

[project.entry-points."echemistpy.analyzers"]
gcd = "echemistpy.analysis.echem.galvanostatic:GalvanostaticAnalyzer"
```

内置插件不需要特殊扫描目录；第三方插件通过 `importlib.metadata.entry_points()` 发现。

## 8. CLI 设计

### 8.1 CLI 框架选择

建议使用 `Typer`。

理由：

- 命令层类型标注清晰。
- 比 argparse 更适合多子命令。
- 输出帮助信息友好。
- 适合快速发展阶段。

需要在 `pyproject.toml` 增加：

```toml
[project.scripts]
echem = "echemistpy.cli.app:app"

[project.optional-dependencies]
cli = ["typer>=0.12", "rich>=13"]
```

如果希望避免新依赖，也可以用 argparse，但长期命令树会更难维护。

### 8.2 命令树

```text
echem
  formats
  inspect PATH
  load PATH
  export PATH
  analyze PATH
  workflow NAME PATH
  doctor
```

### 8.3 命令职责

`formats`

- 列出已注册 reader。
- 输出 extension、instrument、technique、是否支持目录。

`inspect PATH`

- 只读取元数据和基本数据结构。
- 输出 dataset dims、variables、coords、attrs、reader 匹配结果。
- 不做重分析。

`load PATH`

- 读取并标准化。
- 默认输出摘要。
- 可选 `--out raw.nc/json/csv`。

`export PATH`

- 读取后导出为 `nc/csv/json`。
- 支持 `--no-standardize`。

`analyze PATH`

- 读取、标准化、选择 analyzer、运行分析、保存结果。
- 支持 `--params params.yaml`。

`workflow NAME PATH`

- 用于 operando XAS 等多步流程。
- workflow 不应挤在 `scripts/`，应作为正式模块。

`doctor`

- 检查依赖、插件、可导入性、测试环境、可选依赖缺失。

### 8.4 输出规则

CLI 默认输出人类可读摘要：

```text
Reader: BiologicMptReader
Instrument: biologic
Technique: echem,gcd
Rows: 12000
Variables: time_s, ewe_v, current_ma, capacity_mah
Output: raw.nc
```

同时提供机器可读输出：

```bash
echem inspect data.mpt --json
```

错误输出必须明确：

```text
Error: no reader found for .xlsx
Available xlsx readers:
  - lanhe
Suggestion: pass --instrument lanhe
```

## 9. 服务层 API

CLI 应调用 service，不应直接调用 reader/analyzer 内部方法。

建议 API：

```python
from echemistpy.io.api import load_data, inspect_data, list_formats
from echemistpy.analysis.api import analyze_data

bundle = load_data(path, instrument="biologic", standardize=True)
result = analyze_data(bundle, technique="gcd")
```

服务层职责：

- reader 匹配
- option 合并
- strict/lenient 模式
- warning 收集
- 标准化
- 保存 provenance

## 10. 依赖拆分

建议最小依赖：

```text
core:
  numpy
  pandas
  xarray
  h5netcdf
  openpyxl

cli:
  typer
  rich

analysis:
  scipy
  matplotlib
  scikit-learn
  lmfit

xas:
  xraylarch

stxm:
  scikit-image
  umap-learn
  numba

dev:
  ruff
  pytest
  pytest-cov
  pre-commit
  ty
```

这样用户可以：

```bash
pip install echemistpy-cli[cli]
pip install echemistpy-cli[xas]
pip install echemistpy-cli[stxm]
pip install echemistpy-cli[all]
```

## 11. 测试策略

第一阶段测试只追求可运行和契约稳定：

- `test_imports.py`: 所有公开模块可导入。
- `test_formats.py`: 内置 reader 注册成功。
- `test_reader_contracts.py`: 每个 reader 有 `spec`，扩展名唯一或可按 instrument disambiguate。
- `test_standard_names.py`: 标准化后变量名符合 schema。
- `test_cli_smoke.py`: `echem formats`, `echem doctor` 可运行。
- `test_analysis_registry.py`: 默认 analyzer registry 可实例化。

每个仪器格式需要最小 fixture：

```text
tests/fixtures/
  biologic/sample.mpt
  lanhe/sample.xlsx
  claess/sample.dat
  mistral/sample.hdf5
  mspd/sample.xye
```

如果真实样本有版权或体积问题，应制作极小 synthetic fixture。

## 12. 迁移路线

### 阶段 0：冻结当前行为

目标：确认当前行为和失败点。

工作：

- 增加 `tests/test_imports.py`。
- 增加 `tests/test_io_formats.py`。
- 记录当前失败项，不急于修所有科学逻辑。

验收：

- `echemistpy.io.list_supported_formats()` 可测试。
- `AnalysisPipeline()` 当前失败被测试捕获，作为后续修复目标。

### 阶段 1：修复包结构和可导入性

目标：项目能作为 Python 库稳定导入。

工作：

- 给 `analysis/`, `analysis/echem/`, `analysis/stxm/`, `analysis/xas/` 增加 `__init__.py`。
- 修复 `echem/analyzer.py` 的 `from .registry` 为正确导入。
- 修复 `stxm/analyzer.py` 的旧路径 `echemistpy.processing...`。
- 修复 `create_default_registry()` 导入路径。
- 暂时移除或隔离 `scripts/workflow_operando.py` 的旧 `claess2025` 引用。

验收：

- `AnalysisPipeline().registry.available()` 成功。
- 公开 import 全部通过。

### 阶段 2：建立标准数据契约

目标：reader、standardizer、analyzer 使用同一套变量名。

工作：

- 新增 `io/standard_names.py`。
- 重写 `column_mappings.py` 输出目标名。
- 修复 XAS 标准名冲突。
- 明确 `DataTree` 节点级 metadata 存放规则。

验收：

- 每个 reader 加载后可通过 schema 校验。
- analyzer required variables 和标准化输出一致。

### 阶段 3：重构 I/O 插件

目标：reader 可扩展、可测试、可被 CLI 查询。

工作：

- 新增 `ReaderSpec`。
- reader 类声明 `spec`。
- registry 不再依赖类名推断。
- 保留旧 `load()` 作为兼容包装。
- 把目录加载通用逻辑抽到 service，reader 只负责单文件解析，除非确实需要仪器特殊目录逻辑。

验收：

- `list_formats()` 返回结构化 registry 信息。
- 同扩展名多 reader 时必须按 instrument disambiguate。

### 阶段 4：新增 CLI 薄壳

目标：最小 CLI 可用。

工作：

- 新增 `echemistpy.cli.app`。
- 增加 `[project.scripts] echem = ...`。
- 实现 `formats`, `doctor`, `inspect`, `load`, `export`。
- CLI 调用 service API，不直接调用 reader 内部。

验收：

- `echem formats` 可运行。
- `echem inspect sample.mpt` 可运行。
- `echem export sample.mpt --format csv --out sample.csv` 可运行。

### 阶段 5：分析与 workflow

目标：CLI 支持可配置分析。

工作：

- 新增 `AnalyzerSpec`。
- 修复默认 registry。
- `analyze` 支持 `--technique`, `--instrument`, `--params yaml/json`。
- 把 operando XAS 示例从 `scripts/` 迁移到 `workflows/operando_xas.py`。

验收：

- `echem analyze sample.mpt --technique gcd` 可运行。
- `echem workflow operando-xas ./data --config config.yaml` 有明确输入/输出契约。

## 13. 风险与取舍

- `traitlets` 是否保留：如果项目未来需要 notebook/widget 配置联动，保留有价值；如果目标是 CLI 和脚本，`dataclasses` 或 `pydantic` 更直接。
- `DataTree` 是否作为公共契约：它适合层级数据，但对普通 CLI 用户复杂。CLI 输出应隐藏复杂性，只在 `inspect --verbose` 展示树结构。
- 重型依赖拆分会改变安装体验，但这是 CLI 实用性的关键。
- 标准名重命名会破坏旧 API，需要提供兼容层或迁移说明。
- 科学算法不应在结构重构中大改，否则难以判断问题来自架构还是算法。

## 14. 优先级建议

最高优先级：

- 修包导入和 registry。
- 增加 CLI 入口。
- 固定标准变量名。
- 增加 smoke tests。

中优先级：

- reader spec 化。
- 依赖 extras 拆分。
- `doctor` 命令。
- 输出 JSON 摘要。

低优先级：

- 完整 workflow CLI。
- 第三方 entry point 插件。
- 高级可视化命令。
- 性能优化和并行加载。

## 15. 推荐第一批改动清单

如果进入实际编码，建议第一批 PR 只做“可运行骨架”，不要同时大改算法：

1. 新增 package `__init__.py` 并修复错误导入。
2. 新增 `echemistpy/cli/app.py`，实现 `echem formats` 和 `echem doctor`。
3. 在 `pyproject.toml` 添加 `[project.scripts]`。
4. 新增 `tests/test_imports.py` 和 `tests/test_formats.py`。
5. 修复 `scripts/workflow_operando.py` 的旧包名或移动为非发布示例。
6. 增加 `ReaderSpec`，但先不重写所有 reader 内部逻辑。

这样可以快速把项目从“源码原型”推进到“可安装、可调用、可验证”的 CLI 库基础。
