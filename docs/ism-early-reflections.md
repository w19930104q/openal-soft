# Image-Source Early Reflections for openal-soft reverb

## 问题

`AL_EFFECT_REVERB` 的早期反射（ER）只有 4 条固定延迟 tap
（`EARLY_TAP_LENGTHS`，按 5m 平均房间推导），只随 Density 整体缩放，
不随 source/listener 位置变化，也不体现真实房间几何。

## 方案：shoebox image source method（ISM）

矩形房间（边长 Lx/Ly/Lz，listener 在原点，墙位于 ±L/2）。声源的
非居中坐标 s_a = S_a + L_a/2。每个轴有两个 Allen–Berkley 镜像族
（family 位 f_a ∈ {0,1}），镜像索引 p：

    plus 族 (f=0):  img = 2 p L + s - L/2,   轴反射阶 2|p|
    minus 族(f=1):  img = 2 p L - s - L/2,   轴反射阶 2p-1 (p≥1),
                                                  1-2p (p≤0)
    总阶 = 三轴阶之和；阶 0 是真实声源（剔除），阶 > order 剔除
    路径长 d = ||img||              延迟 t = d/c
    墙面增益 g_wall = Π R_a^(轴阶)
    幅度   g = g_wall / d        （距离反比，归一化后乘 ReflectionsGain）
    到达方向 u = img / d          （listener 坐标系，直接做 B-Format 声像）

minus 族的 p=0（近墙）与 p=1（远墙）都是一阶镜像，因此 order=1 恰好
得到 6 个墙面镜像；order=2 再加入最多 18 个边/角镜像（共 24 tap）。
每 tap 独立：分数延迟（线性插值）+ 独立 gain + 一阶低通（墙面/空气
高频吸收）+ 独立到达方向（B-Format pan）。

## 房间几何来源

- 显式配置 `[reverb] room-width/height/depth`（米）；
- 缺省时由 `AL_REVERB_DENSITY` 反推：density_mult = cbrt(Density*1000)，
  基准 5m 平均边长（与 legacy tap 的推导基准一致）。
- 墙面反射系数 `wall-reflect`（默认 0.9）；
- 每路径高频吸收由 `AL_REVERB_AIR_ABSORPTION_GAINHF`（5kHz 增益）
  结合路径长度换算成 one-pole 系数。

## 位置链路（多 source 汇聚问题）

EFX effect slot 是多个 source 共享的，wet 信号在 voice 混合后已丢失
单个 source 的位置。处理：

1. `Voice` 增加 `mWetPos`（listener/head 坐标系声源位置，预归一化，
   即 CalcAttnVoiceParams 里 transform 之后、normalize 之前）。
2. voice 向各 slot 混 wet 时，按该 send 的 W 通道增益加权把 mWetPos
   累积到 `EffectSlotBase::mWetPosAccum/mWetPosWeight`，
   得到 slot 级能量加权"声像中心"。
3. 每个 context 更新周期开始清零累加器；reverb 读取累加结果。
   无活跃 send 时冻结上一次的位置。
4. listener 始终在原点（head-relative），房间坐标系跟随 listener 平移。

## 实时性：为什么 tap 更新必须在 process() 而不是 update()

`EffectState::update()` 只在 effect 属性变更（`slot->Update` 有待处理
props）时由 `CalcEffectSlotParams` 调用；声源移动走的是 voice 参数路径
（`CalcVoiceParams`，每块执行），并不触发 effect slot 的 update()。
若在 update() 里重建 ISM tap，ER pattern 只会在 effect 参数变化时刷新，
无法跟随声源移动（实测移动轨迹下首峰只在个别窗口跳变而非连续移动）。

因此架构上：

- `update()` 仅缓存每块计算所需的 effect 级输入（density 系数、空气
  吸收 LP、ReflectionsDelay、主 MixParams）到 ReverbState 成员；
- `process()` 每块必定执行，在读取 ISM 延迟线之前调用
  `updateImageSource()`，用当前块的累加位置重新生成 tap 目标并做
  one-pole 平滑。这样 ER 延迟/gain/方向逐块跟随声源，且平滑抑制
  zipper noise。cross-fade 期间旧 pipeline 保持其 tap 几何，仅整体
  ISM 输出增益随淡变归零。
- 首次获得有效位置时，tap 几何（延迟/滤波/方向/声像）直接对齐目标值
  而非从零平滑爬升，保证开场脉冲即得到正确的反射时序；gain 仍从零
  淡入以避免爆音。

## 与现有管线的关系

- legacy 4-tap ER 通路完整保留；ISM 开启时其声像输出增益置 0
  （late reverb 的输入仍由 legacy 散射通路提供，保证尾音不变）。
- ISM 是一条独立的 mono 分数延迟线，tap 读出后编码成 B-Format
  （W + 方向 Y/Z/X，N3D），经与 legacy 相同的 pan gain 机制输出。
- 配置 `[reverb] image-source = false` 时行为与原版完全一致
  （默认关闭）。

## 文件改动

- `core/voice.h`        : Voice::mWetPos
- `core/effectslot.h`   : slot 级 wet 位置累加字段
- `alc/alu.cpp`         : 每块清零累加器；CalcAttnVoiceParams 计算 mWetPos
- `core/voice.cpp`      : DoMix 中按 send 增益累加位置
- `alc/effects/reverb.cpp` : ISM 引擎、配置读取、管线集成
- `alsoftrc.sample`       : 新增 `[reverb] image-source*` / `room-*` /
  `wall-reflect*` 配置键文档

## 验证（tests/ism，离线 ALC_SOFT_loopback，48kHz 立体声 float）

独立 Python oracle（`ism_reference.py`，与 C++ 相同的双族枚举，不引用
任何 openal-soft 代码）+ `analyze.py` 三项检查：

1. tap 精度：4 个声源位置、order=1 的全部 24 个预测 tap 在渲染 IR 中
   零管线偏移 100% 命中，平均时延残差 0.33–0.74ms（< 2ms 容差）；
   order=2 的 96 个 tap（24×4）同样 100% 命中。
2. 空间感：ISM 的早期峰时间随声源位置变化显著大于 legacy 固定 tap
   （均值移动 1.35ms vs 0.31ms），且立体声 L/R 平衡随镜像到达方向
   正确偏侧（如 dir.x=-0.78 的 tap 偏左约 -6.7dB）。
3. 实时跟踪：周期脉冲 + 逐块移动声源，ISM 相邻 ER 窗口余弦距离
   0.826 vs legacy 0.052（约 15.8×），主导反射峰沿轨迹连续移动，
   而 legacy 几乎不变。

固定位置测试在录制前先空跑 2048 帧（PRIME_FRAMES），让 one-pole
平滑收敛到新位置的稳态，从而度量几何精度而非位置阶跃的收敛瞬态。
运行：`bash run_test.sh` 后 `python3 -I analyze.py`。
