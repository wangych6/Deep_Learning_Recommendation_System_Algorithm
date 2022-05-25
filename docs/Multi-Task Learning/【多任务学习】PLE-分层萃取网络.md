# 【多任务学习】PLE-分层萃取网络

## 总结

1. CGC在结构上设计的分化，实现了专家功能的分化，而PLE则是通过分层叠加，使得不同专家的信息进行融合。整个结构的设计，是为了让多任务学习模型，不仅可以学习到各自任务独有的表征，还能学习不同任务共享的表征。



## Motivation

文章首先提出多任务学习中不可避免的两个缺点：

- 负迁移（Negative Transfer）：针对相关性较差的任务，使用shared-bottom这种硬参数共享的机制会出现负迁移现象，不同任务之间存在冲突时，会导致模型无法有效进行参数的学习，不如对多个任务单独训练。
- 跷跷板现象（Seesaw Phenomenon）：针对相关性较为复杂的场景，通常不可避免出现跷跷板现象。多任务学习模式下，往往能够提升一部分任务的效果，但同时需要牺牲其他任务的效果。即使通过MMOE这种方式减轻负迁移现象，跷跷板问题仍然广泛存在。

在腾讯视频推荐场景下，有两个核心建模任务：

- VCR(View Completion Ratio)：播放完成率，播放时间占视频时长的比例，回归任务
- VTR(View Through Rate) ：有效播放率，播放时间是否超过某个阈值，分类任务

这两个任务之间的关系是复杂的，在应用以往的多任务模型中发现，要想提升VTR准确率，则VCR准确率会下降，反之亦然。

上一小节提到的MMOE网络存在如下几个缺点

- MMOE中所有的Expert是被所有任务所共享，这可能无法捕捉到任务之间更复杂的关系，从而给部分任务带来一定的噪声。
- 在复杂任务机制下，MMOE不同专家在不同任务的权重学的差不多
- 不同的Expert之间没有交互，联合优化的效果有所折扣

## Model

### CGC：每个任务单独一个专家+定制门控

PLE将共享的部分和每个任务特定的部分**显式的分开**，强化任务自身独立特性。把MMOE中提出的Expert分成两种，任务特定task-specific和任务共享task-shared。保证expert“各有所得”，更好的降低了弱相关性任务之间参数共享带来的问题。

网络结构如图所示，同样的特征输入分别送往三类不同的专家模型（任务A专家、任务B专家、任务共享专家），再通过门控机制加权聚合之后输入各自的Tower网络。门控网络，把原始数据和expert网络输出共同作为输入，通过单层全连接网络+softmax激活函数，得到分配给expert的加权权重，与attention机制类型。

![image-20220517203618822](/Users/chester/Desktop/Deep_Learning_Recommender_System /docs/imges/image-20220517203618822.png)

任务A有 ![[公式]](https://www.zhihu.com/equation?tex=m_A) 个expert，任务B有 ![[公式]](https://www.zhihu.com/equation?tex=m_B) 个expert，另外还有 ![[公式]](https://www.zhihu.com/equation?tex=m_S) 个任务A、B共享的Expert。这样对Expert做一个显式的分割，可以让task-specific expert只受自己任务梯度的影响，不会受到其他任务的干扰（每个任务保底有一个独立的网络模型)，而只有task-shared expert才受多个任务的混合梯度影响。

MMOE则是将所有Expert一视同仁，都加权输入到每一个任务的Tower，其中任务之间的关系完全交由gate自身进行学习。虽然MMOE提出的门控机制理论上可以捕捉到任务之间的关系，比如任务A可能与任务B确实无关，则MMOE中gate可以学到，让个别专家对于任务A的权重趋近于0，近似得到PLE中提出的task-specific expert。如果说MMOE是希望让expert网络可以对不同的任务各有所得，则PLE是保证让expert网络各有所得。

### PLE：分层萃取

PLE就是上述CGC网络的多层纵向叠加，以获得更加丰富的表征能力。在分层的机制下，Gate设计成两种类型，使得不同类型Expert信息融合交互。task-share gate融合所有Expert信息，task-specific gate只融合specific expert和share expert。模型结构如图：

![image-20220517204543553](/Users/chester/Desktop/Deep_Learning_Recommender_System /docs/imges/image-20220517204543553.png)

### 训练策略

1. 联合训练

**联合训练比较适合在同一数据集进行训练**，使用同一feature，但是不同任务输出不同结果，比如前面的pctr和pctcvr任务。

最开始上线的版本中，使用联合训练的方式，并且ctr和ysl两个任务的Loss系数都是1，后来考虑到ysl的均值是4左右，而ctr的均值不到0.2，模型会偏向于ysl。但是手动调节权值非常耗时，考虑使用**UWL**(Uncertainty to Weigh Losses)，优化不同任务的权重系数。

![image-20220517224547413](/Users/chester/Desktop/Deep_Learning_Recommender_System /docs/imges/image-20220517224547413.png)

2. Alternative Training在训练任务A时，不会影响任务B的Tower，同样训练任务B不会影响任务A的Tower，这样就避免了如果任务A的Loss降低到很小，训练任务B时影响任务A的Tower，以及学习率的影响。

   **Alternative Training比较适合在不同的数据集上输出多个目标，多个任务不使用相同的feature**，比如WDL模型，Wide侧和Deep侧用的特征不一样，使用的就是Alternative Training，Wide侧用的是FTRL优化器，Deep侧用的是Adagrad或者Adam。

## Coding

```python
def PLE(dnn_feature_columns, shared_expert_num=1, specific_expert_num=1, num_levels=2,
        expert_dnn_hidden_units=(256,), tower_dnn_hidden_units=(64,), gate_dnn_hidden_units=(),
        l2_reg_embedding=0.00001,
        l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
        task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr')):
    """Instantiates the multi level of Customized Gate Control of Progressive Layered Extraction architecture.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param shared_expert_num: integer, number of task-shared experts.
    :param specific_expert_num: integer, number of task-specific experts.
    :param num_levels: integer, number of CGC levels.
    :param expert_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of expert DNN.
    :param tower_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
    :param gate_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of gate DNN.
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN.
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN.
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks
    :return: a Keras model instance.
    """
    num_tasks = len(task_names)
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")

    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of task_types")

    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    # single Extraction Layer
    def cgc_net(inputs, level_name, is_last=False):
        # inputs: [task1, task2, ... taskn, shared task]
        specific_expert_outputs = []
        # build task-specific expert layer
        for i in range(num_tasks):
            for j in range(specific_expert_num):
                expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                                     seed=seed,
                                     name=level_name + 'task_' + task_names[i] + '_expert_specific_' + str(j))(
                    inputs[i])
                specific_expert_outputs.append(expert_network)

        # build task-shared expert layer
        shared_expert_outputs = []
        for k in range(shared_expert_num):
            expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                                 seed=seed,
                                 name=level_name + 'expert_shared_' + str(k))(inputs[-1])
            shared_expert_outputs.append(expert_network)

        # task_specific gate (count = num_tasks)
        cgc_outs = []
        for i in range(num_tasks):
            # concat task-specific expert and task-shared expert
            cur_expert_num = specific_expert_num + shared_expert_num
            # task_specific + task_shared
            cur_experts = specific_expert_outputs[
                          i * specific_expert_num:(i + 1) * specific_expert_num] + shared_expert_outputs

            expert_concat = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(cur_experts)

            # build gate layers
            gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                             seed=seed,
                             name=level_name + 'gate_specific_' + task_names[i])(
                inputs[i])  # gate[i] for task input[i]
            gate_out = tf.keras.layers.Dense(cur_expert_num, use_bias=False, activation='softmax',
                                             name=level_name + 'gate_softmax_specific_' + task_names[i])(gate_input)
            gate_out = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

            # gate multiply the expert
            gate_mul_expert = tf.keras.layers.Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
                                                     name=level_name + 'gate_mul_expert_specific_' + task_names[i])(
                [expert_concat, gate_out])
            cgc_outs.append(gate_mul_expert)

        # task_shared gate, if the level not in last, add one shared gate
        if not is_last:
            cur_expert_num = num_tasks * specific_expert_num + shared_expert_num
            cur_experts = specific_expert_outputs + shared_expert_outputs  # all the expert include task-specific expert and task-shared expert

            expert_concat = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(cur_experts)

            # build gate layers
            gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                             seed=seed,
                             name=level_name + 'gate_shared')(inputs[-1])  # gate for shared task input

            gate_out = tf.keras.layers.Dense(cur_expert_num, use_bias=False, activation='softmax',
                                             name=level_name + 'gate_softmax_shared')(gate_input)
            gate_out = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

            # gate multiply the expert
            gate_mul_expert = tf.keras.layers.Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
                                                     name=level_name + 'gate_mul_expert_shared')(
                [expert_concat, gate_out])

            cgc_outs.append(gate_mul_expert)
        return cgc_outs

    # build Progressive Layered Extraction
    ple_inputs = [dnn_input] * (num_tasks + 1)  # [task1, task2, ... taskn, shared task]
    ple_outputs = []
    for i in range(num_levels):
        if i == num_levels - 1:  # the last level
            ple_outputs = cgc_net(inputs=ple_inputs, level_name='level_' + str(i) + '_', is_last=True)
        else:
            ple_outputs = cgc_net(inputs=ple_inputs, level_name='level_' + str(i) + '_', is_last=False)
            ple_inputs = ple_outputs

    task_outs = []
    for task_type, task_name, ple_out in zip(task_types, task_names, ple_outputs):
        # build tower layer
        tower_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                           name='tower_' + task_name)(ple_out)
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit)
        task_outs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outs)
    return model
```