# Experiment 004

# AirSim无人机Agent控制实验

# 1. 实验基本信息

## 实验日期

2026.8.21

## 实验人员

loveuav02

## 实验目的

在前三次实验的基础上，引入 **Agent（智能体）框架**，探索大语言模型从“根据 Prompt 直接生成无人机控制行为”进一步发展为“根据任务目标自主思考、选择工具并执行任务”的可行性。

本次实验主要验证：

1. `smolagents` Agent 框架能否正常接入大语言模型。
2. Agent 能否理解用户给出的自然语言任务。
3. Agent 能否根据任务自主选择并调用 Tool。
4. 将 AirSim 无人机控制 API 封装为 Agent Tool 是否可行。
5. 为后续无人机自主搜索、目标识别和复杂任务规划建立基础。

本阶段重点验证的核心链路为：

用户自然语言任务

↓

LLM

↓

Agent

↓

任务推理 / 工具选择

↓

Tool

↓

AirSim API

↓

无人机执行动作

---

# 2. 实验背景

Experiment 001 中，通过 Python 直接调用 AirSim API，实现了无人机起飞、飞行、拍照、返航和降落等基础控制。

Experiment 002 中，引入 LLM 和 Prompt Engineering，使大语言模型能够根据用户的自然语言指令生成无人机控制行为。

Experiment 003 中，进一步引入 VLM 和目标检测模型，使无人机获得了一定程度的视觉感知能力，可以通过摄像头图像识别和检测环境中的目标。

但是前三个阶段仍然存在一个核心问题：

**系统主要还是“用户告诉无人机如何执行”，而不是“用户告诉无人机要完成什么，由无人机自己决定如何完成”。**

例如用户提出：

> “帮我找到场景中的可乐，确认目标后返回。”

传统 Prompt 控制需要提前设计较完整的任务流程，例如：

搜索 → 转向 → 拍照 → 检测 → 判断 → 飞行 → 再检测 → 返航。

随着任务复杂度提高，单纯依赖 Prompt 很难提前描述所有可能出现的情况。

因此，本实验尝试引入 Agent。

Agent 的核心思想是：

**LLM 不再只负责生成一次控制指令，而是根据当前任务和环境反馈进行多步推理，并自主决定下一步需要调用什么工具。**

理想情况下，无人机任务流程可以从：

用户

↓

人工设计详细飞行步骤

↓

LLM生成控制代码

↓

AirSim执行

逐渐转变为：

用户提出任务目标

↓

Agent理解任务

↓

Agent制定计划

↓

选择无人机控制 / 感知工具

↓

执行动作

↓

获得环境反馈

↓

继续推理

↓

完成任务

这也是从 **LLM-controlled UAV** 向 **LLM-based UAV Agent** 转变的重要一步。

---

# 3. Agent系统设计

本实验采用 `smolagents` 作为初步 Agent 框架。

Agent主要由三部分组成：

## 3.1 LLM

使用兼容 OpenAI API 的大语言模型接口，通过 `LiteLLMModel` 接入 Agent。

LLM主要负责：

* 理解用户任务
* 分解任务
* 判断需要使用哪些工具
* 根据 Tool 返回结果继续推理
* 决定下一步行为
* 判断任务是否完成

## 3.2 Agent

使用：

`CodeAgent`

作为本阶段实验的 Agent。

Agent 不直接控制无人机，而是通过调用已经注册的工具完成任务。

其逻辑可以表示为：

Task

↓

Reasoning

↓

Tool Selection

↓

Tool Execution

↓

Observation

↓

Reasoning

↓

Next Tool

↓

……

↓

Final Answer

因此，与 Experiment 002 中“大模型直接按照 Prompt 生成控制过程”的模式相比，Agent 增加了一个持续的：

**Reasoning → Action → Observation**

循环。

## 3.3 Tools

为了让 Agent 能够操作无人机，需要将 AirSim API 封装为 Agent 可以理解和调用的 Tool。

本实验对 `airsim_wrapper.py` 进行改写，并使用 `@tool` 装饰器将无人机功能注册为 Agent Tool。

当前主要包括以下几类能力：

### 无人机控制

* `takeoff()`：无人机起飞
* `land()`：无人机降落
* `fly_to()`：飞往指定坐标
* `fly_path()`：沿指定路径飞行
* `set_yaw()`：调整无人机偏航角
* `get_drone_position()`：获取无人机位置和姿态

### 无人机视觉

通过 AirSim 前置摄像头获取 RGB 图像，使 Agent 可以获得环境观测信息。

### 目标检测

将目标检测能力封装为：

`detect(object_names)`

Agent 可以根据任务需要主动调用目标检测模型，例如：

`detect("duck")`

或者：

`detect("cola")`

获取当前摄像头画面中目标的类别和 Bounding Box。

因此，系统逐步形成：

Agent

├── Flight Tools
│　├── takeoff
│　├── land
│　├── fly_to
│　├── fly_path
│　└── set_yaw
│
├── State Tools
│　└── get_drone_position
│
└── Vision Tools
　　├── camera
　　└── detect

---

# 4. 实验步骤

## Step 1：安装并测试 Agent 框架

首先安装并导入 `smolagents`：

主要使用：

* `CodeAgent`
* `LiteLLMModel`
* `tool`

配置大语言模型后，建立最基础的 Agent。

本阶段首先没有直接连接无人机，而是设计简单 Tool 验证 Agent 是否能够完成：

**自然语言理解 → 工具选择 → 工具调用 → 返回结果**

这一基础闭环。

---

## Step 2：编写简单 Tool 进行 Agent 调用测试

编写测试工具：

`hello(your_name)`

并将其注册到 `CodeAgent`。

给 Agent 输入自然语言任务后，由 Agent 自主分析任务，并选择调用 `hello()` Tool。

实验中可以观察到 Agent 的执行过程，包括：

Task

↓

Step 1 Reasoning

↓

生成工具调用代码

↓

执行 Tool

↓

获得 Tool 返回结果

↓

生成 Final Answer

说明：

**LLM → Agent → Tool**

基础调用链路已经能够正常工作。

这一实验虽然没有直接控制无人机，但验证了后续 AirSim Tool 接入 Agent 的技术路线是可行的。

---

## Step 3：改写 AirSim Wrapper

在原有 `airsim_wrapper.py` 基础上进一步改写，使无人机控制函数符合 `smolagents` Tool 格式。

通过 `@tool` 装饰器将原有 AirSim 控制能力暴露给 Agent。

例如：

自然语言任务：

“让无人机起飞。”

Agent理论执行过程：

用户任务

↓

Agent分析任务

↓

发现需要调用 `takeoff`

↓

`takeoff()`

↓

AirSim API

↓

无人机起飞

因此，Agent 不需要知道 AirSim 底层：

`enableApiControl()`

`armDisarm()`

`takeoffAsync()`

等具体 API 实现。

Agent只需要理解：

`takeoff = 起飞无人机`

即可完成高层任务到低层控制接口之间的连接。

---

## Step 4：接入视觉检测 Tool

为了让 Agent 不仅能够控制无人机，还能够根据环境反馈做出决策，将目标检测模型封装为 Tool：

`detect(object_names)`

其工作流程为：

AirSim Camera

↓

获取当前 RGB 图像

↓

目标检测模型

↓

Object Class + Bounding Box

↓

返回 Agent

↓

Agent根据结果决定下一步动作

例如执行自主搜索任务：

“找到场景中的可乐。”

未来 Agent 可以执行：

起飞

↓

观察环境

↓

`detect("cola")`

↓

没有检测到目标

↓

调整无人机方向

↓

继续飞行

↓

再次检测

↓

发现目标

↓

靠近目标

↓

确认目标

↓

返航

从而形成一个真正具有闭环特征的智能无人机系统。

---

# 5. 实验结果

本阶段已经完成：

1. `smolagents` Agent 框架接入。
2. 大语言模型通过 `LiteLLMModel` 接入 Agent。
3. `CodeAgent` 基础运行流程验证。
4. Agent 自主选择并调用简单 Tool 的流程验证。
5. AirSim Wrapper 向 Agent Tool 格式的改造。
6. 无人机飞行控制接口 Tool 化。
7. 无人机状态获取接口 Tool 化。
8. 视觉目标检测接口 Tool 化。

目前验证结果说明：

**Agent 框架可以作为 LLM 与 AirSim 无人机控制系统之间的新一层任务决策模块。**

系统架构由此前的：

User

↓

Prompt

↓

LLM

↓

AirSim API

逐渐演变为：

User

↓

Task

↓

LLM Agent

↓

Planning / Reasoning

↓

Tool Selection

↓

Flight Tool / Vision Tool

↓

AirSim

↓

Environment

↓

Observation

↓

Agent

这一结构已经具备基础的 Agent 闭环思想。

但是，本阶段主要完成的是：

**Agent 框架及 Tool Calling 控制链路验证。**

完整的无人机自主搜索任务仍需要进一步实验和验证。

因此，本次实验不能认为已经实现完整意义上的自主无人机 Agent，而是完成了从 Prompt/VLM 控制向 Agent 控制架构迁移的第一步。

---

# 6. 实验思考

Experiment 001 到 Experiment 004 的实验路线可以总结为：

## Experiment 001：API Control

Python

↓

AirSim API

↓

UAV

解决：

**“程序能不能控制无人机？”**

---

## Experiment 002：Prompt Control

User

↓

LLM

↓

AirSim API

↓

UAV

解决：

**“人能不能通过自然语言控制无人机？”**

---

## Experiment 003：Vision Control

User

↓

LLM / VLM

↓

Camera

↓

Detection

↓

AirSim API

↓

UAV

解决：

**“无人机能不能看见环境中的目标？”**

---

## Experiment 004：Agent Control

User

↓

Task

↓

Agent

↓

Reasoning

↓

Tool

↓

Observation

↓

Reasoning

↓

Action

↓

UAV

希望解决：

**“无人机能不能根据任务目标，自己决定下一步应该做什么？”**

这是四次实验中最大的架构变化。

过去大语言模型更多承担的是：

**自然语言 → 控制指令转换器**

而 Agent 架构中，大语言模型开始承担：

**任务理解 + 规划 + 工具选择 + 状态判断 + 决策**

因此，无人机开始从：

Command-driven UAV

向：

Task-driven UAV Agent

发展。

---

# 7. 当前存在的问题

虽然 Agent 架构提高了系统自主性，但目前仍然存在较多问题。

## 7.1 Agent并不真正理解三维环境

Agent目前获得的环境信息仍然非常有限。

例如目标检测模型只能提供：

* 目标类别
* Bounding Box

但是 Agent 并不知道：

* 目标距离无人机多远
* 目标真实三维坐标
* 目标是否被障碍物遮挡
* 无人机与目标之间是否存在障碍物
* 当前飞行路径是否安全

因此：

**2D视觉检测结果还没有转换成无人机可以直接使用的3D世界信息。**

---

## 7.2 缺少完整的自主搜索策略

假设用户输入：

“找到一只小黄鸭。”

Agent即使知道可以调用 `detect("duck")`，仍然需要解决：

如果当前画面不存在 Duck，下一步应该往哪里飞？

当前系统缺少：

* Search Pattern
* Exploration Strategy
* Coverage Planning
* Frontier Exploration
* 地图记忆

因此容易出现：

随机移动

↓

检测

↓

没有目标

↓

再次随机移动

这种效率较低的搜索方式。

---

## 7.3 Agent缺少长期记忆

在长任务中，无人机需要知道：

“哪些地方已经搜索过？”

“刚才在哪里发现过目标？”

“哪些方向不需要再次探索？”

如果没有 Memory，Agent可能重复搜索同一区域。

因此后续需要引入：

**Agent Memory / World State**

维护任务执行过程中的环境状态。

---

## 7.4 LLM推理速度影响无人机实时性

Agent每执行一步通常需要：

Observation

↓

LLM Reasoning

↓

Tool Call

↓

Observation

↓

LLM Reasoning

因此一个复杂任务可能产生大量 LLM 调用。

对于无人机这种实时系统，大模型推理延迟可能成为重要问题。

因此未来应该把：

高层任务规划

和

低层实时控制

进行分离。

例如：

LLM Agent

↓

高层任务规划

↓

Navigation / Control Module

↓

高频控制无人机

而不是让 LLM 直接参与每一个低层飞行动作。


## 7.5 Agent决策可靠性仍需进一步验证

本实验主要完成了 Agent 与 AirSim 控制接口的集成，并验证了 Agent 能够根据自然语言任务自主调用工具。但目前 Agent 的决策能力仍然主要依赖大语言模型的推理结果，其可靠性尚未经过系统评估。

在简单任务中，Agent 可以较稳定地完成工具调用；然而随着任务复杂度增加，例如连续搜索、多目标识别、路径调整等场景，可能出现以下问题：

* 推理结果不稳定，不同运行过程可能产生不同决策；
* 重复调用相同工具，导致任务执行效率下降；
* 对环境状态理解不足，无法充分利用已有观测信息；
* 在长时间任务中容易偏离初始目标，影响整体任务完成质量。

此外，目前 Agent 的推理过程主要依赖语言模型的内部知识，而缺少针对无人机任务的专业约束。因此，其生成的动作序列仍需要进一步结合导航、控制和安全机制进行验证，而不能直接作为真实无人机的执行指令。

因此，如何提高 Agent 决策的一致性、稳定性和可解释性，将是后续研究的重要方向。

---

# 8. 后续实验计划

本实验验证了 Agent 控制框架的可行性，下一阶段将重点围绕"提升无人机自主完成复杂任务的能力"开展研究。

## 8.1 构建自主搜索流程

目前 Agent 已具备调用飞行控制和目标检测工具的能力，但尚未形成完整的自主搜索闭环。

下一阶段计划设计搜索策略，使无人机能够在未知环境中主动探索，并根据目标检测结果动态调整飞行路径，实现：

起飞

↓

区域搜索

↓

目标检测

↓

判断是否发现目标

↓

继续搜索或靠近目标

↓

完成任务

从"能够调用工具"发展为"能够持续完成任务"。

---

## 8.2 引入环境记忆（Memory）

当前 Agent 每一步决策主要依赖当前观测结果，缺少长期状态记忆。

未来计划构建环境记忆模块，对以下信息进行持续维护：

* 已搜索区域；
* 已访问航点；
* 已发现目标位置；
* 当前任务执行状态；
* 历史工具调用记录。

通过引入 Memory，使 Agent 能够避免重复搜索，提高任务执行效率，并支持更长时间、更复杂的自主任务。

---

## 8.3 引入任务规划（Planning）

目前 Agent 更多依赖语言模型逐步推理，每完成一步再决定下一步动作。

未来计划增加独立的 Planning 模块，在任务开始阶段生成整体执行计划，并根据执行情况动态调整。

例如，对于"寻找目标并返回"任务，可以自动规划：

1. 起飞；
2. 按照预设搜索策略巡视区域；
3. 持续检测目标；
4. 发现目标后靠近确认；
5. 返回起飞点并降落。

通过任务规划，可以减少无效推理，提高整体执行效率。

---

## 8.4 增强环境感知能力

目前 Agent 获取的环境信息主要来源于二维图像目标检测。

后续计划进一步融合：

* 深度图（Depth Image）；
* 点云（Point Cloud）；
* 语义分割（Semantic Segmentation）；
* 无人机位姿信息；
* 地图信息。

构建更加完整的环境表示，使 Agent 不仅能够"看到目标"，还能理解目标与环境之间的空间关系，为自主导航提供支持。

---

## 8.5 探索多Agent协同

在单无人机 Agent 验证完成后，可进一步研究多 Agent 协同任务。

例如：

* 多无人机协同搜索；
* 多区域任务分配；
* 信息共享与协同决策；
* 动态任务重规划。

通过多个 Agent 的协同工作，提高复杂场景下的搜索效率和任务完成能力。

---

## 8.6 面向真实无人机平台迁移

AirSim 提供了良好的仿真环境，但真实无人机系统仍然面临通信延迟、传感器误差、电池限制和环境扰动等问题。

因此，未来还需要研究仿真到真实平台（Sim-to-Real）的迁移方法，包括：

* 控制接口适配；
* 飞行安全约束；
* 实时状态监测；
* 容错机制设计；
* 在线任务调整。

最终目标是在保证安全性的前提下，将 Agent 控制框架逐步应用于真实无人机平台。

---

# 9. 实验总结

本实验完成了 AirSim 与 Agent 框架的初步集成，实现了大语言模型、Agent 与无人机控制接口之间的连接，并验证了 Agent 根据自然语言任务自主选择工具并执行操作的基本流程。

与前三次实验相比，本实验最大的变化在于系统控制方式由"Prompt 驱动"逐步转变为"任务驱动"。Agent 不再仅负责生成一次控制指令，而是在任务执行过程中持续进行推理、选择工具并根据反馈调整行为，为无人机自主完成复杂任务提供了新的实现思路。

虽然当前系统仍处于基础验证阶段，但已经搭建起 Agent 控制无人机的整体框架，为后续引入环境记忆、任务规划、多模态感知和自主决策等能力奠定了基础，也为构建具备自主执行能力的无人机智能体提供了实验依据。

