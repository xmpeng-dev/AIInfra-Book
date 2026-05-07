# GPT-OSS-20B / MI355X — `use_turbo_grouped_mlp` env-var truthy bug fix

**日期**：2026-04-25
**范围**：`/home/xiaompen/mlperf/small_llm_moe_pretraining/primus`（A 仓库）
**状态**：已修复 + 已 200-iter 实跑复核。A 现在跟随 B 走 legacy
`GroupedMLP` 多 hipBLASLt worker stream 路径，不再被困在单 Triton kernel
里把 stream 0 跑满。

## 1. 现象回顾

- A `~/mlperf/.../primus/run.sh` 早期峰值 ~672 TFLOP/s/GPU、step ~1233 ms。
- B `~/mlperf-training/.../primus/run.sh` 早期峰值 ~720 TFLOP/s/GPU、
  step ~1182 ms。
- 同样的代码、同样的 patches、同样的 17 个 `before_train` patch 都成功
  `Apply` 了，args dump 里 `use_turbo_grouped_mlp` 两边**都打印 `False`**。
- 但 A 的 model dump：`(experts): PrimusTurboGroupedMLP()`；
  B 的 model dump：`(experts): GroupedMLP()`。

为什么 args 显示一样、模型选的 experts 类却不一样？

## 2. 根因（YAML × shell × loader 三方踩到的字符串 truthy 坑）

### 2.1 决策点

`primus/backends/megatron/core/extensions/transformer_engine_spec_provider.py`
里的 `grouped_mlp_modules()`：

```python
elif moe_use_grouped_gemm:
    warnings.warn("The legacy GroupedMLP will be deprecated ...")
    return PrimusTurboGroupedMLP if self.cfg.use_turbo_grouped_mlp else GroupedMLP, None
```

A、B 的 `moe_grouped_gemm=True`、`moe_use_legacy_grouped_gemm=True`，都
进 `elif`。最终落到 `if self.cfg.use_turbo_grouped_mlp` —— 直接 Python
真值判断，没做任何字符串规整。

同一文件 `column_parallel_layer_norm_linear()` 里对 `use_turbo_rms_norm`
是这么写的（注意注释）：

```python
# Primus yaml env-var substitution returns the raw string ``"true"`` /
# ``"false"`` (both truthy under plain ``bool(...)``), so coerce
# common string forms explicitly here.
_flag = getattr(self.cfg, "use_turbo_rms_norm", False)
if _flag is True or (
    isinstance(_flag, str) and _flag.strip().lower() in ("true", "1", "yes", "on")
):
    return PrimusTurboLayerNormColumnParallelLinear
```

**Primus 自己早就知道这个坑**，但只在 `use_turbo_rms_norm` 这一处加了
字符串规整，`use_turbo_grouped_mlp` 那一行**没补**。

### 2.2 Primus YAML loader 不识别 `"true"/"false"`

`primus/core/config/yaml_loader.py`：

```python
def replace_match(m):
    var, default = m.group(1), m.group(2)
    if default is None:
        if var not in os.environ:
            raise ValueError(...)
        return os.environ[var]
    return os.environ.get(var, default)        # ${VAR:default}
...
return _try_numeric(replaced) if replaced != s else replaced

def _try_numeric(v: str):
    if re.fullmatch(r"-?\d+", v):  return int(v)     # 整数
    if FLOAT_PATTERN.fullmatch(v): return float(v)   # 浮点
    return v                                          # 其它一律保持字符串
```

`_try_numeric` 只识整数 / 浮点，**不识 `"true"/"false"`**，env 替换出来
是裸字符串。容器里实测：

```bash
$ export USE_TURBO_GROUPED_MLP=False
$ python3 -c "from primus.core.config.yaml_loader import _resolve_env_in_string; \
            v=_resolve_env_in_string('${USE_TURBO_GROUPED_MLP:true}'); \
            print(type(v).__name__, repr(v), bool(v))"
str 'False' True
```

### 2.3 A 触发 / B 绕开

A 仓库（trace.yaml + 老版生产 yaml 都用的写法）：

```yaml
use_turbo_grouped_mlp: ${USE_TURBO_GROUPED_MLP:true}
```

A 的 `config_*.sh`：

```bash
export USE_TURBO_GROUPED_MLP=False
```

→ env 替换成字符串 `"False"` → `bool("False") == True` →
**PrimusTurboGroupedMLP**。意图"关掉"实际是"开着"。

B 仓库的 yaml：

```yaml
use_turbo_grouped_mlp: false       # 硬编码，YAML 直接解析成 Python False
```

→ Python `False` → **GroupedMLP**。同样的 bug 因为 yaml 走的是字面量
而不是 env 替换路径，**绕开了**。

而 args dump（`arguments.py:1198`）用 `f"{val}"` 格式化，
`f"{False}"` 和 `f"{'False'}"` 的输出**完全一样**，所以从日志看不出来
—— 这就是这个 bug 一直藏着的原因。

### 2.4 实测的生产 yaml 又比这个更糙

正式的 `gpt_oss_20B-pretrain-fp8.yaml`（不是 trace.yaml）A 仓库本来就是

```yaml
use_turbo_grouped_mlp: true        # 直接硬编码 true
```

跟 shell 的 `export USE_TURBO_GROUPED_MLP=False` 互相**完全无视**，
shell 那行在生产里是纯噪声，但很容易让人误以为已经关掉了。

## 3. 为什么 `GroupedMLP` 自带"多流并行"

B 选到的是 Megatron legacy 路径 `gg.ops.gmm`，
最终落到 `grouped_gemm==1.1.4` 的 C++ 扩展 `grouped_gemm_backend.so`。
对它做 `nm -CD` 看到这些符号：

```text
U c10::hip::getCurrentHIPStream(signed char)
U c10::hip::HIPStream::stream() const
U hipEventCreate@hip_4.2
U hipEventRecord@hip_4.2
U hipStreamCreateWithFlags@hip_4.2
U hipStreamWaitEvent@hip_4.2
U hipblasCreate
U hipblasGemmEx
U hipblasSetStream
```

也就是说这个扩展自己 `hipStreamCreateWithFlags` 创建一组 worker stream，
用 `hipEventRecord/hipStreamWaitEvent` 跟主 stream 同步，然后**逐 expert
调用 `hipblasGemmEx`，每个 expert 落在不同 worker stream 上**。trace 里
B 多出来的 stream 13/14/15/16 就是它生出来的。

A 那条 Triton GroupedGEMM 路径只是单 kernel，所有 expert 平铺在一颗
Triton kernel 内，**只能跑在调用它的当前 stream**（stream 0）。

## 4. 修复

### 4.1 改了 4 个文件

| 文件 | 改动 |
|------|------|
| `mlperf/.../primus/gpt_oss_20B-pretrain-fp8.yaml` | `use_turbo_grouped_mlp: true` → `false`，并在原地写明原因 + 实测 step ms |
| `mlperf/.../primus/config_MI355X_1x8x1_tp1pp1ep1_gbs32_fp8.sh` | 删掉 `export USE_TURBO_GROUPED_MLP=False` 这行误导性的 export，加 NOTE 提醒以后**不要再加回来** |
| `mlperf/.../primus/gpt_oss_20B-pretrain-fp8.trace.yaml` | 同步硬编码 `false`，避免下次 trace 复测又踩 env 替换 |
| `workspace/Primus-mlperf/{primus,pp}/backends/megatron/core/extensions/transformer_engine_spec_provider.py` | 给 `use_turbo_grouped_mlp` 那一行加上和 `use_turbo_rms_norm` 完全一致的字符串 coerce（防御性） |

### 4.2 spec provider 防御性 patch

```python
elif moe_use_grouped_gemm:
    warnings.warn("The legacy GroupedMLP will be deprecated ...")
    # Primus yaml env-var substitution returns the raw string ``"true"`` /
    # ``"false"`` (both truthy under plain ``bool(...)``), so coerce common
    # string forms explicitly here. Mirror the same handling used in
    # ``column_parallel_layer_norm_linear`` for ``use_turbo_rms_norm``.
    _flag = getattr(self.cfg, "use_turbo_grouped_mlp", False)
    use_turbo = _flag is True or (
        isinstance(_flag, str) and _flag.strip().lower() in ("true", "1", "yes", "on")
    )
    return (PrimusTurboGroupedMLP if use_turbo else GroupedMLP), None
```

真值表（off-host smoke 已过）：

| 输入                                            | 选中             |
|------------------------------------------------|------------------|
| `True / "true" / "True" / "TRUE" / " True " / "1" / "yes" / "on"` | PrimusTurboGroupedMLP |
| `False / "false" / "False" / "FALSE" / "0" / "no" / "off" / "" / None` | GroupedMLP |

关键的字符串 `"False"` / `"false"` 现在**会被识别成关闭**，再不会被
truthy 误判。

## 5. 实跑复测（200 iter，gbs=32 / EP=1 / FP8）

修复后在容器里 `bash run.sh` 直接复跑：

- model dump：`(experts): GroupedMLP()` ✓
- 头一段（warmup 后 step 20–90）：

```text
iter= 20  ms=1130.4/1564.6  TFLOPS=730.6/571.9
iter= 30  ms=1124.4/1417.9  TFLOPS=734.5/626.1
iter= 50  ms=1116.2/1581.6  TFLOPS=739.8/588.7
iter= 60  ms=1123.3/1505.2  TFLOPS=735.2/613.1
iter= 70  ms=1147.1/1454.1  TFLOPS=719.9/628.4
iter= 80  ms=1167.4/1418.2  TFLOPS=707.4/638.2
iter= 90  ms=1181.3/1391.9  TFLOPS=699.1/645.0
```

- 长稳段（step 110–190，已度过初期 warmup）：

```text
iter=110  ms=1194.0/1194.0  TFLOPS=691.6/691.6   ← rolling reset
iter=130  ms=1228.8/1225.5  TFLOPS=672.0/674.1
iter=150  ms=1203.9/1226.0  TFLOPS=686.0/673.9
iter=170  ms=1207.2/1220.4  TFLOPS=684.1/676.9
iter=190  ms=1208.1/1223.8  TFLOPS=683.6/675.1
```

修复前同一脚本（A 仓库历史 `run.log` 里同步号步）：

```text
iter= 30  ms=1230.1/1531.5  TFLOPS=671.4/576.0
iter= 50  ms=1222.1/1693.4  TFLOPS=675.7/543.1
iter= 60  ms=1226.7/1615.6  TFLOPS=673.2/564.8
iter= 90  ms=1295.4/1495.9  TFLOPS=637.5/595.7
iter=110  ms=1240.8/1240.8  TFLOPS=665.6/665.6
iter=120  ms=1293.0/1266.9  TFLOPS=638.7/652.1
```

汇总：

| 阶段              | 修复前 instant | 修复后 instant | Δ        |
|-------------------|----------------|----------------|----------|
| step 20–90 中段   | 638 – 675      | **699 – 740**  | +8 ~ +10 % |
| step 110+ 长稳    | 638 – 666      | 654 – 692      | +1 ~ +4 %  |
| 单步最佳          | 675            | **739.8**      | +9.6 %     |
| 单步最快 ms       | 1222           | **1116**       | −8.7 %     |

长稳段提升缩小到 ~2%，但峰值和中段都很扎实，且**主流不再被 turbo
Triton kernel 占满**，给上层进一步压重叠（NCCL grad-reduce、tier-1
elementwise 融合）腾出了空间——这是后续 B 系列优化的前提。

## 6. 经验

1. Primus 的 `${VAR:default}` 是**字符串**替换，不是 YAML/Python 真值。
   一旦 yaml 里一个布尔型字段写成 `${VAR:true}`，就**绝对不要**在 shell
   里 `export VAR=False`/`export VAR=True`，否则永远是 truthy。要么用空
   字符串 `export VAR=`，要么 `unset VAR`，要么干脆 yaml 里硬编码。
2. args dump（`arguments.py:1198`）的 `f"{val}"` 没法区分布尔 `False` 和
   字符串 `"False"`。要确认布尔型 flag 真的传到位，要看模型 dump 里
   实例化的类（这次是 `(experts): ...`），不要看 args dump。
3. spec provider 那一处只 fix 了 `use_turbo_rms_norm` 没 fix
   `use_turbo_grouped_mlp` 是个**已知漏网之鱼**，本次顺手补齐；后续如
   果再加新的 turbo flag，要对照 `use_turbo_rms_norm` 那段抄一遍 coerce。
4. 在 AMD 上 EP=1 / 32 expert 时，**legacy `grouped_gemm` C++ 扩展 ≫
   PrimusTurbo Triton GroupedGEMM**——前者把 expert 摊到 4 条 hipBLASLt
   worker stream 上并行；后者塞进单颗 Triton kernel 把主 stream 跑满。
   `enable_primus_turbo: true` **不等于**所有 turbo 子 flag 都该开，
   `use_turbo_grouped_mlp` 在这个配置下应当固定 `false`。

## 7. 关联 / 后续

- 关联：`2026-04-21_gptoss_10_moe_fp8_grouped_gemm_diagnosis.md`、
  `2026-04-21_gptoss_11_moe_fp8_GG_actual_run_postmortem.md` 之前对
  grouped GEMM 的诊断，没追到 yaml 替换这一层。
- 关联：`2026-04-24_gptoss_15_ep1_trace_optimization_plan.md` 里的
  trace 计划提到主流被 grouped_gemm 占满 —— 这次是其根因。
- 后续：往 Primus 上游提一个 PR，把 spec provider 那两处的 coerce
  统一抽成一个 `_truthy(...)` helper；同时给 yaml_loader 加可选的
  `bool` 识别（最少在 `${VAR:true}` / `${VAR:false}` 的情况下识别），
  避免别的项目再踩。
- 后续：B 系列接着做 `compute/NCCL overlap`（stream 0 现在有 ~46% 空
  闲，可以塞 NCCL grad-reduce）和 tier-1 elementwise 融合（fused
  residual + RMSNorm V1/V2 已经实现，待全量收敛后默认打开）。
