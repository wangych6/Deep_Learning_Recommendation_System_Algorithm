# 【多任务学习】 ESMM--全空间多任务建模

## 总结

1. ESMM首创了利用用户行为序列数据在完整样本空间建模，并提出利用学习CTR和CTCVR的辅助任务，迂回学习CVR，避免了传统CVR模型经常遭遇的样本选择偏差和训练数据稀疏的问题，取得了显著的效果。
2. 使用全量样本的效果好于负采样，我分析原因是因为能让模型见过更多的负样本，学习更加充分。这种情况下如果正负样本不均衡，可以选择对正样本重新采样copy进数据集，或者修改loss，对正负样本预测错误增加不同的惩罚。
3. Loss只使用了ctr的loss和ctcvr的loss，不用cvr的loss是因为cvr的样本只是在子空间中的，存在SSB问题，而ctr和ctcvr是在全量样本空间采集的。这个模型的motivation就是跳过优化cvr，只把cvr作为中间变量来运用。



## Motivation

传统的CVR预估问题存在着两个主要的问题：**样本选择偏差**和**稀疏数据**。下图的白色背景是曝光数据，灰色背景是点击行为数据，黑色背景是购买行为数据。传统CVR预估使用的训练样本仅为灰色和黑色的数据。

<img src="/Users/chester/Desktop/Deep_Learning_Recommender_System /docs/imges/image-20220517170122284.png" alt="image-20220517170122284" style="zoom:150%;" />

这会导致两个问题：

- 样本选择偏差（sample selection bias，SSB）：如图所示，CVR模型的正负样本集合={点击后未转化的负样本+点击后转化的正样本}，但是线上预测的时候是样本一旦曝光，就需要预测出CVR和CTR以排序，样本集合={曝光的样本}。构建的训练样本集相当于是从一个与真实分布不一致的分布中采样得到的，这一定程度上违背了机器学习中训练数据和测试数据独立同分布的假设。
- 训练数据稀疏（data sparsity，DS）：点击样本只占整个曝光样本的很小一部分，而转化样本又只占点击样本的很小一部分。如果只用点击后的数据训练CVR模型，可用的样本将极其稀疏。

阿里妈妈团队提出ESMM，借鉴多任务学习的思路，引入两个辅助任务CTR、CTCVR(已点击然后转化)，同时消除以上两个问题。

三个预测任务如下：

- **pCTR**：p(click=1 | impression)；
- **pCVR**: p(conversion=1 | click=1,impression)；
- **pCTCVR**: p(conversion=1, click=1 | impression) = p(click=1 | impression) * p(conversion=1 | click=1, impression)；

> 注意：其中只有CTR和CVR的label都同时为1时，CTCVR的label才是正样本1。如果出现CTR=0，CVR=1的样本，则为不合法样本，需删除。 pCTCVR是指，当用户已经点击的前提下，用户会购买的概率；pCVR是指如果用户点击了，会购买的概率。

三个任务之间的关系为：

![image-20220517170418125](/Users/chester/Desktop/Deep_Learning_Recommender_System /docs/imges/image-20220517170418125.png)

其中x表示曝光，y表示点击，z表示转化。针对这三个任务，设计了如图所示的模型结构：

![image-20220517170523608](/Users/chester/Desktop/Deep_Learning_Recommender_System /docs/imges/image-20220517170523608.png)

如图，主任务和辅助任务共享特征，不同任务输出层使用不同的网络，将cvr的预测值*ctr的预测值作为ctcvr任务的预测值，利用ctcvr和ctr的label构造损失函数：

![image-20220517170702885](/Users/chester/Desktop/Deep_Learning_Recommender_System /docs/imges/image-20220517170702885.png)

从公式中可以看出，pCVR 可以由pCTR 和pCTCVR推导出。从原理上来说，相当于分别单独训练两个模型拟合出pCTR 和pCTCVR，再通过pCTCVR 除以pCTR 得到最终的拟合目标pCVR 。在训练过程中，模型只需要预测pCTCVR和pCTR，利用两种相加组成的联合loss更新参数。pCVR 只是一个中间变量。而pCTCVR和pCTR的数据是在完整样本空间中提取的，从而相当于pCVR也是在整个曝光样本空间中建模。

- 提供特征表达的迁移学习（embedding层共享）。CVR和CTR任务的两个子网络共享embedding层，网络的embedding层把大规模稀疏的输入数据映射到低维的表示向量，该层的参数占了整个网络参数的绝大部分，需要大量的训练样本才能充分学习得到。由于CTR任务的训练样本量要大大超过CVR任务的训练样本量，ESMM模型中特征表示共享的机制能够使得CVR子任务也能够从只有展现没有点击的样本中学习，从而能够极大地有利于缓解训练数据稀疏性问题。

模型训练完成后，可以同时预测cvr、ctr、ctcvr三个指标，线上根据实际需求进行融合或者只采用此模型得到的cvr预估值。

## Experiments

![image-20220517172054389](/Users/chester/Desktop/Deep_Learning_Recommender_System /docs/imges/image-20220517172054389.png)

- 负采样 vs 过采样：AMAN是对负样本采样，Over Sampling是对正样本进行copy后的，可以看到oversampling效果会更好。我分析就是让模型见过了更多的负样本。
- 除法 vs 乘法：除法通过预测pctr和pctcvr来得到cvr，对比ESMM效果差一些，原因是因为预测pctcvr的数值都特别小，处以一个很小的数值的波动会特别大。
- 不共享Embedding：效果略差于ESMM。

## Coding

关键步骤都写了注释

```python
def ESSM(dnn_feature_columns, task_type='binary', task_names=['ctr', 'ctcvr'],
         tower_dnn_units_lists=[[128, 128],[128, 128]], l2_reg_embedding=0.00001, l2_reg_dnn=0,
         seed=1024, dnn_dropout=0,dnn_activation='relu', dnn_use_bn=False):

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,seed)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    ctr_output = DNN(tower_dnn_units_lists[0], dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input) # ctr塔
    cvr_output = DNN(tower_dnn_units_lists[1], dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input) # cvr塔

    ctr_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(ctr_output)
    cvr_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(cvr_output)

    ctr_pred = PredictionLayer(task_type, name=task_names[0])(ctr_logit) 
    cvr_pred = PredictionLayer(task_type)(cvr_logit)

    ctcvr_pred = tf.keras.layers.Multiply(name=task_names[1])([ctr_pred, cvr_pred])# 两者直接相乘 CTCVR = CTR * CVR

    model = tf.keras.models.Model(inputs=inputs_list, outputs=[ctr_pred, cvr_pred, ctcvr_pred])
    return model
```