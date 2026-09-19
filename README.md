# BasketballAnnotation

BasketballAnnotation 是一个独立的篮球视频标注与数据转换项目。它不依赖、读取或修改 TimeSoccer 源码；两者之间唯一的预期接口是本项目导出的标准 JSON。

当前阶段只建立最小数据工作流，不包含 Qwen、Llama、翻译模型、GPU 推理、GUI、数据库、Docker 或任务调度系统。

## 目录职责

- `data/annotations/`：保存权威单视频标注。人工事实、人工确认后的中文解说和最终英文解说只在这里保存。
- `schemas/`：保存权威标注的 JSON Schema。
- `scripts/`：保存只使用 Python 标准库的数据转换脚本。
- `tests/`：保存 exporter 和数据约束测试。

未经人工确认的模型候选以后应保存到独立的 `data/preannotations/` 层，不能直接写入权威 annotation。本轮没有创建该目录。

## 权威时间轴

`video` 是权威时间轴所依据的完整源视频。每个事件的 `timestamp_sec` 必须始终相对于该源视频的起点，而不是相对于送给模型观察的临时片段。

例如，原始视频第 `617.0` 秒发生事件。即使模型观看的是原视频 `600.0` 到 `660.0` 秒的临时片段，并报告局部时间 `17.0` 秒，写入权威 annotation 时也必须恢复为：

```text
timestamp_sec = 617.0
```

临时 observation clip 的文件名和局部时间不写入权威 annotation。当前 `basketball_001.mp4` 本身就是完整的 60 秒源视频，因此当前时间戳保持为 `8.0`、`12.0`、`17.0` 等值。

所有事件必须满足：

```text
0.0 <= timestamp_sec <= duration_sec
```

## 事件字段与状态

每个事件包含：

- `event_id`：单视频内唯一且稳定的内部引用编号，例如 `E001`。不得进入 TimeSoccer 训练答案。插入或删除事件时应尽量保持已有 ID 不变。
- `timestamp_sec`：相对于完整源视频起点的绝对秒数。
- `fact_zh`：人工确认的原始中文事实。模型和转换脚本不得改写。
- `commentary_zh`：经过人工确认的中文解说。未经审核的模型草稿不能写入此字段。
- `commentary_en`：最终用于 TimeSoccer 训练答案的英文解说。
- `status`：权威事件当前所处阶段。
- `notes`：人工审核备注或不确定项。

只使用以下三个状态：

- `fact_confirmed`：人工事实已确认，两个 commentary 字段仍为空。
- `commentary_confirmed`：中文 commentary 已经人工确认，英文 commentary 仍为空。
- `ready_for_export`：中英文 commentary 均已完成，可以正式导出。

当前不使用 `event_type`。未来即使添加，它也只能是辅助 metadata，不能默认进入 TimeSoccer 训练答案。

如果以后确认一条人工记录实际包含两个事件，应由人工为拆分后的事件指定各自的时间和 `event_id`。不得由模型或 exporter 自动拆分。

## 基本导出

本轮 exporter 只支持将一个完整源视频的 annotation 导出为一个 TimeSoccer JSON 条目：

```powershell
python scripts\export_timesoccer.py `
  --input data\annotations\basketball_001.annotation.json `
  --output exports\basketball_001.timesoccer.json
```

只有所有事件均为 `ready_for_export` 且 `commentary_en` 非空时才允许导出。当前示例的英文 commentary 为空，因此 exporter 应明确拒绝正式导出。

导出的训练答案只包含：

```text
timestamp + commentary_en
```

`event_id`、`fact_zh`、`commentary_zh`、`status`、`notes` 和 `source_game_id` 均不会进入训练答案。时间始终以 decimal 形式输出，例如 `17.0 seconds`。

exporter 不会修改输入 annotation，并拒绝输入与输出指向同一文件。

## 未来的训练窗口

权威 annotation 始终保存源视频绝对时间。未来生成 1-minute、3-minute 或 5-minute 训练窗口时，必须在派生阶段计算局部时间：

```text
relative_timestamp = absolute_timestamp - window_start_sec
```

其中 `absolute_timestamp` 是 annotation 中的 `timestamp_sec`。本轮不实现窗口切片系统，当前 exporter 等价于使用 `window_start_sec = 0.0` 的完整源视频窗口。

train/test 划分以后必须按 `source_game_id` 进行，不能把同一场原始比赛的不同片段随机分到 train 和 test。

## 运行测试

测试只使用 Python 标准库：

```powershell
python -m unittest discover -s tests -v
```

