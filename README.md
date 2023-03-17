# MARL
多智能体强化学习VDN、QMIX、QTRAN、QPLEX复现，主要参考了pymarl和原始论文提供的源码
# main训练伪代码
![image](https://user-images.githubusercontent.com/78296375/225800123-ee80dddd-c30f-4eb6-affc-9d0ef56bc31e.png)
几个关键概念的作用：

MultiAgentController()：
- 包含了多智能体的网络和所需要的操作
- 用于生成agent模型（value-based算法为q net；AC的为actor net critic net）
- 包含agent的操作，如产生action、计算individual q值等
SMAC()：

- 星际争霸多智能体强化学习环境
- 可以编写自己的环境，最好根据SMAC提供的一套api封装一下
ReplayBuffer()：

- 用于存数据(注意有两种存储方式：1）存transition；2）存episode。根据算法需求选择)
- 用于sample数据，喂给模型训练

Learner：
- 学习器，充当replaybuffer和mac的粘合剂，即用某种算法使用buffer中的数据训练mac

rollout：
- 主要有两个参数：mac、env
- 作用是mac和env交互，生成数据然后存放到buffer中
- 注意rollout调用mac中的网络是evaluate模式，不需要梯度，只用用来生成数据
# MARL
多智能体强化学习VDN、QMIX、QTRAN、QPLEX复现，主要参考了pymarl和原始论文提供的源码
# main训练伪代码
![image](https://user-images.githubusercontent.com/78296375/225800123-ee80dddd-c30f-4eb6-affc-9d0ef56bc31e.png)
几个关键概念的作用：

MultiAgentController()：
- 包含了多智能体的网络和所需要的操作
- 用于生成agent模型（value-based算法为q net；AC的为actor net critic net）
- 包含agent的操作，如产生action、计算individual q值等
SMAC()：

- 星际争霸多智能体强化学习环境
- 可以编写自己的环境，最好根据SMAC提供的一套api封装一下
ReplayBuffer()：

- 用于存数据(注意有两种存储方式：1）存transition；2）存episode。根据算法需求选择)
- 用于sample数据，喂给模型训练

Learner：
- 学习器，充当replaybuffer和mac的粘合剂，即用某种算法使用buffer中的数据训练mac

rollout：
- 主要有两个参数：mac、env
- 作用是mac和env交互，生成数据然后存放到buffer中
- 注意rollout调用mac中的网络是evaluate模式，不需要梯度，只用用来生成数据

# MARL
多智能体强化学习VDN、QMIX、QTRAN、QPLEX复现，主要参考了pymarl和原始论文提供的源码
# main训练伪代码
![image](https://user-images.githubusercontent.com/78296375/225800123-ee80dddd-c30f-4eb6-affc-9d0ef56bc31e.png)
几个关键概念的作用：

MultiAgentController()：
- 包含了多智能体的网络和所需要的操作
- 用于生成agent模型（value-based算法为q net；AC的为actor net critic net）
- 包含agent的操作，如产生action、计算individual q值等
SMAC()：

- 星际争霸多智能体强化学习环境
- 可以编写自己的环境，最好根据SMAC提供的一套api封装一下
ReplayBuffer()：

- 用于存数据(注意有两种存储方式：1）存transition；2）存episode。根据算法需求选择)
- 用于sample数据，喂给模型训练

Learner：
- 学习器，充当replaybuffer和mac的粘合剂，即用某种算法使用buffer中的数据训练mac

rollout：
- 主要有两个参数：mac、env
- 作用是mac和env交互，生成数据然后存放到buffer中
- 注意rollout调用mac中的网络是evaluate模式，不需要梯度，只用用来生成数据
