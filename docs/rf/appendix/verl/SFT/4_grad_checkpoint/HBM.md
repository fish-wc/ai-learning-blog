# HBM（High Bandwidth Memory，高带宽显存）
## 一句话
**HBM就是GPU的显存（VRAM），和普通显卡GDDR显存是同类东西，但架构完全不一样，带宽极高，多用于A100/H100这类AI训练卡。**

> 你前面FlashAttention里提到的：减少HBM读写，就是减少**显存和芯片之间的数据搬运**。

## 简单对比 HBM vs GDDR
- **HBM**：堆叠在GPU芯片旁边，靠硅中介层互联。带宽巨大，但单位容量成本贵。A100 40G/80G、H100 80G都是HBM。
- **GDDR（GDDR6/GDDR6X）**：独立显存颗粒，通过PCB走线连GPU。消费卡RTX4090、3090用的就是GDDR，带宽低于HBM。

## 为什么FlashAttention特别关心HBM？
GPU的性能瓶颈**很多时候不是计算单元（SM）不够快，而是HBM的数据读写太慢**。
- HBM：容量大，但是**访问延迟高、读写带宽是瓶颈**；
- SRAM（片上共享内存/寄存器）：就在GPU计算核心旁边，**带宽极高，但容量极小**。

FlashAttention核心思路：
> 尽量把Q/KV小块放在**高速SRAM**里面完成计算，**不把巨大的N×N注意力分数矩阵写回HBM**，大幅减少HBM读写流量，从而提速+省显存。

## 和你前面几个概念串起来
1. **参数、中间激活，都存在HBM**
2. 朴素Attention：算出完整N×N注意力矩阵 → 存入HBM（巨大显存开销，OOM来源）
3. FlashAttention：分块tile计算，**不往HBM存完整注意力矩阵**
4. 梯度检查点：减少保存在HBM里的中间激活
5. 动态batch：控制总token，防止HBM被一次性塞满OOM

> 通俗比喻：
> SRAM = 计算器上面的草稿纸（很小，随手写擦，速度飞快）
> HBM = 大文件夹柜子（很大，但拿放文件慢）
> 朴素Attention：把整张超大表格全部存进柜子；
> FlashAttention：只每次拿一小块放到草稿纸上算，算完扔掉，不存入柜子。

## 补充两个面试常问小点
1. HBM容量 = 大家常说的“80G A100显存”，这个80G就是HBM容量；
2. HBM带宽单位：GB/s，H100 HBM3带宽接近3TB/s，远高于GDDR。
