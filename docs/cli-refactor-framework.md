# echemistpy-cli CLI 库重构设计框架

日期：2026-06-04

## 1. 结论

`echemistpy-cli` 不应该只是在当前 `io + analysis` 原型上补一个 CLI 入口。当前库里已经有多仪器 reader、`RawData/RawDataInfo`、`xarray.Dataset/DataTree`、标准化映射、分析 registry/pipeline 雏形，这些方向值得保留；但它们的职责边界不够清楚，尤其是：

- `io` 同时承担 reader、标准化、数据容器、保存导出、目录聚合，边界过宽。
- `analysis` 同时包含科学算法和 XAS 绘图函数，结果数据与图形表达混在一起。
- 目前没有独立 `data` 层承载“标准数据契约、xarray 接口、存储、转换、分发、索引/数据库”。
- 目前没有独立 `plot` 层承载“从标准数据/分析结果生成图”的公共接口。
- CLI 层尚不存在，不能直接验证“可安装、可调用、可扩展”的产品形态。

因此推荐的目标不是“把所有功能塞进一个 CLI”，而是做成一个 **CLI-first scientific library**：

```text
CLI 是薄壳
Python API 是稳定核心
io 只负责外部格式适配
data 负责统一数据模型、schema、转换、存储和 xarray/数据库接口
analysis 负责科学计算
plot 负责可视化
workflow 负责编排多步任务
```

这比当前文档中偏 `core/io/analysis` 的方案更贴近项目长期目标，也更符合同类成熟科学库的结构。

## 2. GitHub 参考项目

这次设计参考了几个真实项目的仓库结构，而不是只从当前代码出发。

### 2.1 ixdat

仓库：https://github.com/ixdat/ixdat

相关结构：

- `src/ixdat/readers`
- `src/ixdat/exporters`
- `src/ixdat/plotters`
- `src/ixdat/techniques`
- `src/ixdat/calculators`
- `src/ixdat/backends`
- `src/ixdat/data_series.py`
- `src/ixdat/measurement_base.py`
- `src/ixdat/db.py`

可借鉴点：

- 原位实验数据需要一个稳定的“measurement/data object”作为核心，而不是让 reader/analyzer 直接传散乱对象。
- reader、exporter、plotter、technique/calculator 分开，有助于避免 I/O、算法、绘图互相依赖。
- 数据库/backend 是一等模块，适合管理实验数据的持久化与检索。

对 echemistpy 的启发：

- `data` 层应该成为项目中心，不应继续放在 `io.structures`。
- `plot` 不应作为 `analysis.xas.plotting` 的附属物。
- 操作数据库、缓存、索引、批量数据集时，不应通过 reader 或 analyzer 临时处理。

### 2.2 cellpy

仓库：https://github.com/jepegit/cellpy

相关结构：

- `cellpy/cli.py`
- `cellpy/readers`
- `cellpy/exporters`
- `cellpy/filters`
- `cellpy/internals`
- `cellpy/parameters`
- `cellpy/utils`
- `setup.py` 中配置 `cellpy=cellpy.cli:cli`

可借鉴点：

- 电池/电化学数据处理确实需要正式 CLI，而不是只给 notebook 使用。
- reader、exporter、参数、过滤工具分开是实用结构。
- CLI 入口应是明确发布契约。

需要避免的点：

- `cellpy/cli.py` 是一个很大的单文件 CLI。echemistpy 不应复制这种结构，应该从第一版就拆成 `cli/app.py + cli/commands/*`。
- cellpy 的主依赖比较重。echemistpy 应该把 CLI、基础 I/O、XAS、TXM/STXM、plot、dev 依赖拆成 extras。

### 2.3 HyperSpy

仓库：https://github.com/hyperspy/hyperspy

相关结构：

- `hyperspy/api.py`
- `hyperspy/signal.py`
- `hyperspy/axes.py`
- `hyperspy/model.py`
- `hyperspy/drawing`
- `hyperspy/learn`
- `hyperspy/io.py`
- `hyperspy/hyperspy_extension.yaml`
- `pyproject.toml` 中使用 optional dependencies，例如 `learning`, `speed`, `gui-jupyter`, `gui-traitsui`

可借鉴点：

- 成熟科学库通常有一个面向用户的公开 API 文件，不让用户直接依赖内部模块树。
- 数据对象、坐标轴、模型、绘图、学习/分析能力分层明显。
- 扩展能力用声明式 metadata 管理，比靠类名推断更可靠。
- 可选依赖拆分很重要，重型能力不应污染基础安装。

对 echemistpy 的启发：

- 可以提供 `echemistpy.api` 作为 notebook/脚本入口，CLI 调用同一组 service。
- reader/analyzer/plotter 都应有显式 spec，而不是目录扫描后靠类名猜测。
- XAS/TXM/STXM 等重型依赖必须放入 extras。

### 2.4 RosettaSciIO

仓库：https://github.com/hyperspy/rosettasciio

相关结构：

- `rsciio/_io_plugins.py`
- 每个格式目录有自己的模块与 `specifications.yaml`
- `rsciio/hspy`, `rsciio/nexus`, `rsciio/netcdf`, `rsciio/tiff`, `rsciio/bruker` 等大量格式目录

可借鉴点：

- I/O 库可以专注于“读写科学文件格式”，不承担分析和绘图。
- 每个 I/O 插件用 `specifications.yaml` 描述 name、alias、description、extension、write support、API module 等能力。
- 插件 registry 应读 metadata，而不是实例化 reader 或通过类名推断。

对 echemistpy 的启发：

- `io` 层应改为“外部格式适配器层”，只输出标准中间数据。
- 当前 `_initialize_default_plugins()` 的类名/扩展名推断应替换为显式 `ReaderSpec`。

### 2.5 pyFAI

仓库：https://github.com/silx-kit/pyFAI

相关结构：

- `src/pyFAI/app`
- `src/pyFAI/io`
- `src/pyFAI/containers.py`
- `src/pyFAI/units.py`
- `src/pyFAI/method_registry.py`
- `src/pyFAI/worker.py`
- `pyproject.toml` 中有多个 `[project.scripts]`

可借鉴点：

- CLI 应放在单独 `app`/`cli` 层，核心算法和 worker/service 不应依赖命令行参数。
- 容器、单位、I/O、方法 registry 都是正式模块。
- 对于成熟命令行工具，可以暴露多个 console scripts。

对 echemistpy 的启发：

- 第一阶段建议只暴露一个 `echem` 入口，内部用子命令管理复杂度；后续若某些 workflow 稳定，再考虑单独脚本。
- `units` 和 `schema` 应在 `data` 层集中管理，不能散落在标准化器、reader 和 analyzer 中。

## 3. 当前代码状态

### 3.1 当前模块

```text
src/echemistpy/
  __init__.py
  cli/
    __init__.py
    app.py
    main.py
    commands/
      __init__.py
      convert_data.py
      doctor.py
      formats.py
      inspect_data.py
  data/
    __init__.py
    column_mappings.py
    models.py
    schema.py
    standardize.py
    storage.py
    utils.py
  analysis/
    __init__.py
    registry.py
    pipeline.py
    echem/__init__.py
    echem/analyzer.py
    stxm/__init__.py
    stxm/analyzer.py
    xas/__init__.py
    xas/analyzer.py
    xas/processing.py
    xas/fitting.py
    xas/plotting.py
    xas/elements.py
  io/
    __init__.py
    base_reader.py
    contracts.py
    conversion.py
    loaders.py
    plugin_manager.py
    summary.py
    plugins/
      __init__.py
      echem_biologic_mpr.py
      echem_biologic_mpt.py
      echem_lanhe_ccs.py
      echem_lanhe_xlsx.py
      TXM_MISTRAL.py
      XAS_CLAESS.py
      XRD_MSPD.py
Samples/
docs/
tests/
```

注意：当前仓库没有 `scripts/workflow_operando.py`。旧文档里对该脚本的描述已经过时。

### 3.2 当前主要问题

- 已有 `[project.scripts] echemistpy = "echemistpy.cli.app:main"`，并已实现 `echem/xas/xrd/txm` 领域子命令下的 `formats`, `inspect`, `convert`，以及顶层 `doctor`。
- `data` 层已经成为真实实现层：`models`, `schema`, `column_mappings`, `standardize`, `storage`, `utils` 都在 `data` 下。
- `io` 层不再保留 `structures`, `standardizer`, `saver`, `column_mappings`, `reader_utils` 兼容 re-export；旧路径应视为内部重构中删除。
- 当前依赖方向应保持为 `io -> data`、`analysis -> data`、`cli -> io/data`，不允许 `data -> io`。
- `analysis.xas.plotting` 应迁到 `plot.xas`，避免分析模块承担绘图接口。
- `analysis/echem`, `analysis/stxm`, `analysis/xas` 已有包导出，分析模块已改为从 `data.models` 消费数据容器。
- XAS/STXM 已开始统一到 `energy_ev/absorption/optical_density` 等标准名，但更完整的 schema validation 尚未实现。
- reader 插件仍依赖插件目录扫描；能力声明已改为 `ReaderSpec`，后续可迁到 entry points。
- 目录聚合逻辑放在 `BaseReader`，会让所有 reader 被迫承担目录组织职责。
- 核心依赖包含 `xraylarch`, `umap-learn`, `numba`, `scikit-image` 等重型包，基础 CLI 用户不应默认安装。
- `tests/` 已覆盖 CLI、reader spec、公开 data API 和电化学样例；完整测试可通过，但样例测试仍偏慢，且 traitlets 初始化有 deprecation warning。

## 4. 目标架构

推荐目标结构：

```text
src/echemistpy/
  api.py
  cli/
    __init__.py
    app.py
    output.py
    errors.py
    commands/
      formats.py
      inspect.py
      convert.py
      index.py
      analyze.py
      plot.py
      workflow.py
      doctor.py
  core/
    exceptions.py
    logging.py
    options.py
    paths.py
  data/
    __init__.py
    models.py
    metadata.py
    schema.py
    units.py
    standardize.py
    transform.py
    xarray.py
    storage.py
    index.py
    validation.py
  io/
    __init__.py
    api.py
    contracts.py
    registry.py
    readers/
      biologic_mpt.py
      lanhe_xlsx.py
      lanhe_ccs.py
      claess_dat.py
      mistral_hdf5.py
      mspd_xye.py
    writers/
      csv.py
      netcdf.py
      json.py
  analysis/
    __init__.py
    api.py
    contracts.py
    registry.py
    echem/
      galvanostatic.py
      cv.py
      eis.py
    xas/
      analyzer.py
      processing.py
      fitting.py
      elements.py
    txm/
      analyzer.py
    xrd/
      analyzer.py
  plot/
    __init__.py
    api.py
    contracts.py
    registry.py
    styles.py
    echem.py
    xas.py
    xrd.py
    txm.py
  workflows/
    __init__.py
    operando_xas.py
  plugins/
    discovery.py
```

### 4.1 模块边界

`io`

- 负责不同数据接口的读取/写入。
- 输入：外部文件、目录、数据库导出、仪器格式。
- 输出：`data.DataBundle`，或者对外写出 CSV/NetCDF/JSON 等。
- 不负责科学分析。
- 不负责绘图。
- 不持有标准变量名的最终解释权，只引用 `data.schema`。

`data`

- 负责中间数据如何传输、转化、分发。
- 负责标准 schema、单位、metadata、provenance、warnings。
- 负责 xarray 接口、DataTree 规则、表格转换、lazy/chunk 策略。
- 负责高效存储：NetCDF/Zarr/HDF5/Parquet，以及 SQLite/DuckDB 索引。
- 负责 cache/index，不把数据库逻辑塞进 reader。

`analysis`

- 负责分析技术：GCD、CV、EIS、XAS normalization/LCF/PCA、TXM/STXM clustering、XRD peak fitting 等。
- 输入：标准化后的 `DataBundle`。
- 输出：`AnalysisBundle` 或 `DataBundle` 的分析结果版本。
- 不读取原始文件。
- 不保存图。
- 不直接处理 CLI 参数。

`plot`

- 负责画图接口。
- 输入：标准数据或分析结果。
- 输出：figure 对象和图像文件。
- 不做改变科学结果的分析，只做视图层需要的轻量选择、组合、样式。
- `analysis.xas.plotting` 应迁入这里。

`cli`

- 只负责命令行参数、配置文件、日志、退出码、人类可读/JSON 输出。
- 所有业务逻辑调用 `api.py` 或各层 service。

`workflows`

- 组合 `io -> data -> analysis -> plot -> storage`。
- 只放稳定、有明确输入输出契约的多步流程。

## 5. 数据层设计

### 5.1 核心对象

第一版不需要发明复杂继承体系。建议用清晰 dataclass 包装 xarray：

```python
@dataclass
class DataBundle:
    data: xr.Dataset | xr.DataTree
    meta: Metadata
    schema: str
    provenance: Provenance
    warnings: list[str] = field(default_factory=list)

@dataclass
class AnalysisBundle:
    data: xr.Dataset | xr.DataTree
    meta: Metadata
    schema: str
    provenance: Provenance
    parameters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
```

现有 `RawData`, `RawDataInfo`, `AnalysisData`, `AnalysisDataInfo` 可以先保留为兼容层，但新代码应逐步转向 `DataBundle/AnalysisBundle`。

### 5.2 xarray 规则

- `xr.Dataset` 用于单个测量或单个分析结果。
- `xr.DataTree` 用于目录、批量、operando 多源数据。
- 坐标和变量命名必须 NetCDF/Zarr 友好，不允许 `/`, `%`, `-` 开头、空格等。
- 单位不要只写在变量名中，应写入 `DataArray.attrs["units"]`，变量名保留物理量与标准单位提示。
- `record` 是通用一维采样维度。
- `cycle_number`, `energy_ev`, `two_theta_deg`, `x_um`, `y_um` 等应优先作为 coord。

### 5.3 标准 schema

第一版建议定义 `schema_version = "echemistpy-raw-v1"` 和 `analysis_schema_version = "echemistpy-analysis-v1"`。

推荐变量名：

```text
通用:
  record
  time_s
  timestamp

电化学:
  cycle_number
  step_number
  voltage_v
  ewe_v
  ece_v
  current_ma
  capacity_mah
  specific_capacity_mah_g
  frequency_hz
  re_z_ohm
  neg_im_z_ohm

XAS:
  energy_ev
  absorption
  norm_absorption
  e0_ev
  edge_step

XRD:
  two_theta_deg
  intensity
  intensity_error
  d_spacing_angstrom

TXM/STXM:
  energy_ev
  x_um
  y_um
  transmission
  optical_density
  cluster_label
```

旧变量名兼容策略：

- reader 可以先输出旧变量名。
- `data.standardize` 统一转换到 schema 名。
- analyzer 只能声明和消费标准名。
- CLI 默认输出标准名；`inspect --raw` 可以显示原始列名映射。

### 5.4 元数据

元数据分组：

```text
identity:
  sample_name
  technique
  instrument
  source_id

acquisition:
  start_time
  operator
  source_path
  file_count

sample:
  active_material_mass_g
  electrolyte
  wavelength_angstrom

provenance:
  reader_name
  reader_version
  schema_version
  source_hash
  created_at
  warnings
  raw_metadata
```

`others` 可以保留为兼容字段，但新模型应使用 `raw_metadata`。

### 5.5 存储与索引

`data.storage` 负责内部高效存储，不等同于 `io.writers`。

推荐规则：

- 单个标准数据：NetCDF (`.nc`) 或 Zarr。
- 大型层级/operando 数据：Zarr 或 HDF5/NetCDF group。
- 表格导出：Parquet/CSV，由 `io.writers` 提供。
- 索引与检索：SQLite 或 DuckDB，只存 metadata、source hash、schema、路径、状态，不直接塞大型数组。
- 缓存 key：`source_hash + reader_name + reader_version + schema_version + options_hash`。

这能支持命令：

```bash
echem index add ./data --instrument biologic --store ./echem-index.duckdb
echem index query --sample MnO2 --technique gcd
echemistpy echem convert data.mpt --out raw.zarr
```

## 6. I/O 层设计

### 6.1 ReaderSpec

借鉴 RosettaSciIO 的声明式插件思路，reader 必须显式声明能力：

```python
@dataclass(frozen=True)
class ReaderSpec:
    name: str
    extensions: tuple[str, ...]
    instruments: tuple[str, ...]
    techniques: tuple[str, ...]
    supports_directory: bool = False
    can_inspect: bool = True
    description: str = ""
```

reader 协议：

```python
class DataReader(Protocol):
    spec: ClassVar[ReaderSpec]

    def inspect(self, path: Path, options: ReaderOptions) -> DataPreview:
        ...

    def read(self, path: Path, options: ReaderOptions) -> DataBundle:
        ...
```

### 6.2 Reader 职责

reader 只做三件事：

1. 识别和解析某种外部格式。
2. 提取原始 metadata。
3. 构造最接近标准 schema 的 `DataBundle`。

reader 不应该：

- 调用 analyzer。
- 生成正式图。
- 管理全局数据库。
- 根据 CLI 参数决定输出格式。
- 依赖全局 plugin manager 反查自己的扩展名。

### 6.3 Registry

内置 reader 可以直接注册：

```python
registry.register(BiologicMptReader)
registry.register(LanheXlsxReader)
registry.register(LanheCcsReader)
```

第三方 reader 后续用 entry points：

```toml
[project.entry-points."echemistpy.readers"]
biologic_mpt = "echemistpy.io.readers.biologic_mpt:BiologicMptReader"
```

同一扩展名多 reader 时：

- `load(path)` 不能静默选第一个。
- 必须返回 ambiguity error，并显示可用 instrument。
- `load(path, instrument="lanhe")` 才能继续。

## 7. 分析层设计

### 7.1 AnalyzerSpec

```python
@dataclass(frozen=True)
class AnalyzerSpec:
    name: str
    domain: str
    techniques: tuple[str, ...]
    instruments: tuple[str, ...] | None
    input_schema: str
    output_schema: str
    required_variables: tuple[str, ...]
```

协议：

```python
class Analyzer(Protocol):
    spec: ClassVar[AnalyzerSpec]

    def analyze(self, bundle: DataBundle, options: AnalysisOptions) -> AnalysisBundle:
        ...
```

### 7.2 分析模块内部规则

- `analysis.echem` 放 GCD/CV/EIS/CA/CP 等。
- `analysis.xas` 放 normalization、E0、LCF、PCA、AutoBK、FFT 等科学计算。
- `analysis.txm` 或 `analysis.stxm` 放图像堆栈配准、PCA、clustering、ROI、chemical map。
- `analysis.xrd` 放 peak detection、拟合、相位/峰表处理。
- 绘图函数不放在 analysis 内。
- 文件读取不放在 analysis 内。
- 每个 analyzer 的 required variables 必须和 `data.schema` 一致。

### 7.3 当前分析层修复方向

- 增加 `analysis/__init__.py`。
- 增加 `analysis/echem/__init__.py`、`analysis/xas/__init__.py`、`analysis/stxm/__init__.py`。
- 修复 `echem/analyzer.py` 的导入为 `from echemistpy.analysis.registry import TechniqueAnalyzer`。
- 修复 `stxm/analyzer.py` 的旧导入路径。
- 让 `create_default_registry()` 从明确模块导入，不依赖子包根的隐式导出。
- XAS analyzer 的 `required_columns` 改为 `("energy_ev", "absorption")`，并同步处理内部变量名。

## 8. Plot 层设计

### 8.1 PlotSpec

```python
@dataclass(frozen=True)
class PlotSpec:
    name: str
    domain: str
    input_schema: str
    required_variables: tuple[str, ...]
    output_kinds: tuple[str, ...] = ("figure", "png", "svg", "pdf")
```

协议：

```python
class Plotter(Protocol):
    spec: ClassVar[PlotSpec]

    def render(self, bundle: DataBundle | AnalysisBundle, options: PlotOptions) -> PlotResult:
        ...
```

### 8.2 推荐 plot API

```python
from echemistpy.plot import plot_data

fig = plot_data(bundle, kind="echem-cycle")
fig = plot_data(result, kind="xas-normalized")
```

CLI：

```bash
echemistpy echem plot data.mpt --kind echem-cycle --out cycle.png
echemistpy xas plot xas.nc --kind xas-normalized --out xas.svg
echemistpy txm plot txm.zarr --kind txm-map --variable optical_density --out map.png
```

### 8.3 绘图边界

plot 可以：

- 选择变量。
- 组合子图。
- 设置样式。
- 输出图像。

plot 不应该：

- 改变分析结果。
- 在内部重新读取原始文件。
- 修改 `DataBundle`。
- 为了画图执行不可见的科学计算；如果需要计算，应先由 analysis 产生结果。

## 9. CLI 设计

### 9.1 入口

第一阶段建议只暴露一个 console script：

```toml
[project.scripts]
echemistpy = "echemistpy.cli.app:main"
```

使用 Typer：

```toml
[project.optional-dependencies]
cli = ["typer>=0.12", "rich>=13"]
```

### 9.2 命令树

```text
echemistpy
  doctor
  echem
    formats
    inspect PATH
    convert PATH
    analyze PATH
    plot PATH
  xas
    formats
    inspect PATH
    convert PATH
    analyze PATH
    plot PATH
  xrd
    formats
    inspect PATH
    convert PATH
    analyze PATH
    plot PATH
  txm
    formats
    inspect PATH
    convert PATH
    analyze PATH
    plot PATH
  index add PATH
  index query
  workflow NAME PATH
```

`echem` 只承载电化学命令，不能列出 XAS/XRD/TXM reader。XAS、XRD、TXM 等材料表征格式必须通过对应领域子命令访问，例如 `echemistpy xas formats`。

### 9.3 命令职责

`formats`

- 列出当前领域的 reader/writer/analyzer/plotter registry。
- 输出 extension、instrument、technique、是否支持目录、是否支持写出。

`inspect PATH`

- 只做 reader 匹配、metadata、dims、变量、coords、schema preview。
- 支持 `--json` 和 `--raw`。

`convert PATH`

- 读取并标准化。
- 输出 NetCDF/Zarr/CSV/Parquet/JSON。
- 取代旧文档里的 `load/export` 命令分裂，减少 CLI 面。

`analyze PATH`

- 读取或打开标准数据。
- 选择 analyzer。
- 保存分析结果。
- 支持 `--check` 做 schema dry-run。

`plot PATH`

- 打开标准数据或分析结果。
- 选择 plotter。
- 保存图。

`index`

- 管理本地数据索引和缓存。

`workflow`

- 编排 operando XAS 等多步流程。

`doctor`

- 检查包导入、插件、可选依赖、schema、测试环境。

### 9.4 CLI 输出

默认人类可读：

```text
Reader: BiologicMptReader
Instrument: biologic
Technique: echem,gcd
Schema: echemistpy-raw-v1
Rows: 12000
Variables: time_s, voltage_v, current_ma, capacity_mah
Warnings: 1
```

机器可读：

```bash
echemistpy echem inspect data.mpt --json
```

错误应可操作：

```text
Error: multiple readers match extension ".xlsx"
Available instruments:
  - lanhe
Suggestion: pass --instrument lanhe
```

## 10. Public Python API

提供 `echemistpy.api`，让 notebook、脚本和 CLI 调用同一套服务。

```python
from echemistpy.api import load_data, analyze_data, plot_data, save_data

bundle = load_data("data.mpt", instrument="biologic")
result = analyze_data(bundle, technique="gcd")
figure = plot_data(result, kind="echem-cycle")
save_data(result, "analysis.nc")
```

服务层职责：

- reader/analyzer/plotter 匹配。
- options 合并。
- strict/lenient 模式。
- warnings 收集。
- schema 校验。
- provenance 写入。
- 输出路径与缓存策略。

## 11. 依赖拆分

当前依赖太重。建议：

```text
base:
  numpy
  pandas
  xarray
  h5netcdf
  openpyxl
  traitlets 或 pydantic/dataclasses 二选一

cli:
  typer
  rich

storage:
  zarr
  pyarrow
  duckdb

plot:
  matplotlib

analysis:
  scipy
  scikit-learn
  lmfit

xas:
  xraylarch

txm:
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

安装示例：

```bash
pip install echemistpy-cli[cli]
pip install echemistpy-cli[cli,plot]
pip install echemistpy-cli[xas]
pip install echemistpy-cli[txm]
pip install echemistpy-cli[all]
```

## 12. 迁移路线

### 阶段 0：记录当前事实

目标：不继续在错误假设上设计。

工作：

- 更新本文档。
- 增加最小导入测试。
- 增加当前 reader registry smoke test。
- 记录当前失败项。

验收：

- `import echemistpy` 通过。
- `from echemistpy.io import list_supported_formats` 通过。
- 文档中的当前模块清单与仓库一致。

### 阶段 1：建立 data 层

目标：把数据契约从 `io` 拆出来，并让 `data` 成为唯一数据实现层。

工作：

- 新增 `echemistpy/data`。
- 物理迁移 `io.structures` 到 `data.models`。
- 物理迁移 `io.standardizer` 到 `data.standardize`。
- 物理迁移 `io.column_mappings` 到 `data.column_mappings`，标准名仍由 `data.schema` 统一声明。
- 物理迁移 `io.saver` 到 `data.storage`。
- 物理迁移 `io.reader_utils` 中与 xarray 名称、metadata 合并、标准 attrs 相关的通用逻辑到 `data.utils`。
- 新增标准名和单位表。
- 删除旧 `io.structures`, `io.standardizer`, `io.saver`, `io.column_mappings`, `io.reader_utils` 文件，不保留兼容 re-export。

验收：

- `src/echemistpy/data` 内没有任何 `echemistpy.io` import。
- `io` reader 和 `analysis` analyzer 直接从 `data.models` 消费数据容器。
- reader 输出可以通过 `data.validation.validate_schema()`。
- analyzer required variables 与 schema 一致。

### 阶段 2：重构 I/O registry

目标：reader 可发现、可测试、可解释。

工作：

- 新增 `ReaderSpec`。
- 为 Biologic MPT、Lanhe XLSX、Lanhe CCS、CLAESS DAT、MISTRAL HDF5、MSPD XYE 声明 spec。
- 移除类名推断和 `_get_file_extension()` 反查。
- 将目录聚合迁到 service 或明确的 directory reader。
- 增加 `inspect()` API。

验收：

- `list_formats()` 返回结构化信息。
- 同扩展名多 reader 时不会静默选错。
- 每个 reader 有最小 fixture smoke test。

### 阶段 3：修复 analysis 层

目标：分析 registry 可用，分析只消费标准数据。

工作：

- 补齐 `__init__.py`。
- 修复错误导入。
- 新增 `AnalyzerSpec`。
- 修复 XAS 标准变量名冲突。
- 将 analyzer 参数整理为 options。

验收：

- `AnalysisPipeline().registry.available()` 成功。
- `echem analyze --check` 能报告 schema 是否满足 analyzer。

### 阶段 4：建立 plot 层

目标：绘图成为独立公共接口。

工作：

- 新增 `echemistpy/plot`。
- 迁移 `analysis/xas/plotting.py` 到 `plot/xas.py`。
- 新增 `plot/echem.py`, `plot/xrd.py`, `plot/txm.py` 的最小接口。
- 新增 `PlotSpec` 和 plot registry。

验收：

- `plot_data(bundle, kind=...)` 返回 matplotlib figure。
- `echemistpy echem plot ... --out figure.png` 可运行。

### 阶段 5：新增 CLI 薄壳

目标：最小 CLI 可发布，CLI 只编排 `io` 和 `data` 服务。

工作：

- 新增 `echemistpy/cli/app.py` 和 `commands/*`。
- 添加 `[project.scripts] echemistpy = "echemistpy.cli.app:main"`。
- 实现 `echemistpy echem formats/inspect/convert`，并为 `xas/xrd/txm` 提供对应领域入口。
- 实现顶层 `doctor`。
- 再实现 `analyze`, `plot`, `index`, `workflow`。

验收：

- `echemistpy echem formats` 只列出电化学 reader。
- `echemistpy xas formats`, `echemistpy xrd formats`, `echemistpy txm formats` 分别列出对应领域 reader。
- `echemistpy echem inspect Samples/...` 可运行。
- `echemistpy echem convert Samples/... --out raw.nc` 可运行。
- `echemistpy doctor` 能报告缺失 optional dependencies。

### 阶段 6：存储、索引和 workflow

目标：支持批量数据和长期复用。

工作：

- 新增 `data.storage` 和 `data.index`。
- 实现 cache key。
- 支持 SQLite/DuckDB 索引。
- 将稳定的 operando XAS 流程迁到 `workflows/operando_xas.py`。

验收：

- `echem index add ./Samples` 可生成索引。
- `echem index query` 可返回匹配数据。
- workflow 有清晰输入、输出、缓存、日志。

## 13. 第一批 PR 建议

第一批不要大改科学算法，目标是先把库变成“可导入、可检查、可验证”的 CLI 库基础。

1. 新增并清理 `data` 包，物理迁移模型、标准化、列名映射、存储和通用工具，不保留 `io` 兼容层。
2. 收紧 `io` 包，只保留 reader contracts、registry、load/inspect/convert 和插件，不从 `io.__init__` 暴露 data 容器或 storage。
3. 新增 `data.schema`，统一变量名，先修 XAS/STXM `energy_ev/absorption/optical_density` 冲突。
4. 补齐 `analysis` 子包导出并让 analyzer 直接依赖 `data.models`。
5. 新增 `ReaderSpec`，先为现有 reader 声明 spec，不重写 reader 内部解析。
6. 新增 `cli/app.py`，实现 `formats`, `inspect`, `convert`, `doctor`。
7. 新增 `tests/test_cli.py`, `tests/test_reader_specs.py`, `tests/test_public_api.py`, `tests/test_echem_samples.py`。

这样改动足够小，但方向已经对齐最终结构。

## 14. 设计取舍

- 是否保留 traitlets：如果未来重 notebook widget/GUI，保留 traitlets 有意义；如果重点是 CLI 和脚本，`dataclasses` 或 pydantic 更直接。第一阶段可以先兼容 traitlets，不在同一 PR 内替换。
- 是否直接采用 HyperSpy/RosettaSciIO：不建议。echemistpy 的范围横跨电化学、XAS、XRD、TXM，且已有 xarray 数据模型；直接依赖大型框架会增加迁移成本。应借鉴它们的边界和 registry 思路。
- 是否把所有写出都放在 `io`：对外格式写出放 `io.writers`，内部缓存/索引/高效存储放 `data.storage`。这能避免 `io` 再次膨胀。
- 是否一开始支持复杂数据库：不建议。先用文件存储 + SQLite/DuckDB metadata index，避免把数据库设计变成主任务。
- 是否一开始暴露多个 CLI scripts：不建议。先用一个 `echem` 子命令入口，等 workflow 稳定再考虑单独脚本。

## 15. 最终判断

更优设计不是“当前 `io` 继续变大、`analysis` 继续加功能、最后补 CLI”，而是先确立中心数据层：

```text
external files
  -> io readers
  -> data schema / standardize / storage
  -> analysis analyzers
  -> plot renderers
  -> cli/workflows as orchestration
```

按这个方向，`echemistpy-cli` 才能同时满足：

- 命令行可用。
- Python API 可复用。
- 新仪器 reader 可扩展。
- 新分析技术可插拔。
- 数据可缓存、可索引、可转换。
- 绘图接口稳定，不污染科学分析逻辑。
