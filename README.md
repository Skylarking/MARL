# MARL
多智能体强化学习VDN、QMIX、QTRAN、QPLEX复现，主要参考了pymarl和原始论文提供的源码
# main训练伪代码
![image](https://user-images.githubusercontent.com/78296375/225800123-ee80dddd-c30f-4eb6-affc-9d0ef56bc31e.png)
几个关键概念的作用：
mac：
- 即Multi-Agent Controller，包含了多智能体所需要的操作
- 用于生成agent模型（value-based算法为q net；AC的为actor net critic net）
- 包含agent的操作，如产生action、计算individual q值等
env：
- 多智能体强化学习环境
- 可以编写自己的环境，可以根据SMAC提供的一套api封装一下
buffer：
- 用于存数据
- 注意有两种存储方式：1）存transition；2）存episode
