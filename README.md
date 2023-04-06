# MARL
多智能体强化学习VDN、QMIX、QTRAN、QPLEX复现，主要参考了pymarl和原始论文提供的源码
# main训练伪代码
![image](https://user-images.githubusercontent.com/78296375/225800123-ee80dddd-c30f-4eb6-affc-9d0ef56bc31e.png)
几个关键概念的作用：

MultiAgentController()：
- 包含了多智能体的网络和所需要的操作
- 用于生成agent模型（value-based算法为q net；AC的为actor net）（不包含mixing net或者critic net）
- 包含agent的操作，如产生action、计算individual q值等

SMAC()：
- 星际争霸多智能体强化学习环境
- 可以编写自己的环境，最好根据SMAC提供的一套api封装一下

ReplayBuffer()：
- 用于存数据(注意有两种存储方式：1）存transition；2）存episode。根据算法需求选择)
- 用于sample数据，喂给模型训练
- 注意：on-policy和off-policy之间的区别。on-policy只能用当前时刻的被改进的policy获取的数据训练，因此训练完成后要清空buffer；off-policy可以使用其他策略（行为策略）采集的数据，不用清空buffer，但可能某些算法需要重要性采样。（MAPPO有些特殊，它是on-policy的算法，但是它可以使用被改进策略某个邻域内的策略所产生的数据训练，可以去看PPO原始论文）

Learner：
- 学习器，充当replaybuffer和mac的粘合剂，即用某种算法使用buffer中的数据训练mac
- 注意：在CTDE范式下，Leaner包含mixing net或者critic Net，因为这两个只是用来辅助我们训练q net或者actor net，训练完成后就丢弃了。只需要q net或者actor net做决策

rollout：
- 主要有两个参数：mac、env
- 作用是mac和env交互，生成数据然后存放到buffer中
- 注意：rollout调用mac中的网络是evaluate模式，不需要梯度，只用来生成数据
