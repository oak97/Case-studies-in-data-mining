---
typora-root-url: ./result
---

[TOC]

### 一  文件介绍

MG1915027卢文婷（基于用户的协同过滤方法）

MG1915010步纤屿（基于物品的协同过滤方法）

MG1915046朱倩雯（基于加权二部图的SlopeOne方法）

文档由三人共同完成

```shell
├── data（原始数据）
│   ├── movies.csv
│   ├── tags.csv
│   ├── testing_ratings.csv
│   └── training_ratings.csv
├── model（算法）
│   ├── explore.ipynb（查看和预处理数据）
│   └── model.ipynb（算法实现）
├── readme.md
└── result（预测结果和中间结果）
    ├── all_pred_testing_ratings.csv（预测结果）
    ├── new_movie.csv（处理了电影类别特征）
    └── train.csv
```



### 二  基于加权二部图的SlopeOne

#### 1  原理

- **SlopeOne**

  对于一对待预测用户和待预测电影，首先找出当前待预测用户已打分电影集合，然后对集合中每部电影进行以下计算：计算给集合中这部电影打过分的用户中有哪些是同时给这部电影和待预测电影同时打过分的，求出打分之差，并对所有这些同时打过两个分的用户求打分差的均值，将该值作为集合中这部电影与待预测电影的评分差值。那么根据一个差值就可以求出一个待预测电影的得分，最后对所有差值得到的预测得分进行平均，求出待预测电影的评分值。

  **然后我们想到，两次均值过程，第一次是公共为两部电影评过分的用户们的差值进行平均，第二次是对已打分电影们进行平均，我们尝试将这两个部分从普通平均变成加权平均。**

- **二部图对最后一步的电影贡献加权**

  下图（来源：[1]）构建了物品-用户的加权二部图，权重是评分，资源a、b、c、d从物品方形先传到用户圆形部分，然后再传回物品方形部分，传播过程中是按照权重来分配资源的，如物品$i_1$的初始资源$a$，因为只有两条边$i_i-u_1$和$i_1-u_2$，再结合边的权重关系，最终2/5a传到$u_1$，3/5a传到$u_2$。经过两次传播，我们可以得到物品内部之间的贡献关系，例如$i_1$的a=47a/150+b/5+c/3+47d/270，那么b、c、d对a的贡献分别是1/5、1/3、47/270，利用这个思路来求当前待预测用户已打分电影们对待预测电影的贡献，并在最后求均值时用上。我们发现这个思路其实和物品相似矩阵很相似。

  ![paper](/paper.png)

  下面的公式求出通过加权二部图资源分配得到的当前集合中已打分电影$j$对待预测电影$i$的贡献程度，其中$U$表示同时给两部电影$i$和$j$评过分的用户集合，$r_{ui}$表示连接节点$u$和$i$的边的权重，$\operatorname{deg}(j)$表示节点$j$的度。
  $$
  \sum_{u \in U} \frac{r_{u i} r_{u j}}{\operatorname{deg}(j) \operatorname{deg}(u)}
  $$

- **用户相似矩阵对一开始的单个电影加权**

  同理，在为两部电影都评过分的用户们的差值进行平均时，考虑每个用户与待预测用户的相似度，进行加权。MAE得分效果提升不明显。

- 均值预测

  本次实验中，若待预测电影没有在训练集出现过，那就用待预测用户以往打分均值作为预测值。

- 调整得分

  因为本数据集的评分只能是0.5-5.0步长为0.5的数值，所以对求出的数值进行近似，变为.0或者.5结尾的小数。

#### 2  代码实现

1. 首先读入训练数据，并根据训练数据构造用户-电影评分矩阵

   ```python
   # 训练集
   train = pd.read_csv("../data/training_ratings.csv",usecols=[0,1,2])
   # 用户-电影评分矩阵
   train_matrix = train.pivot_table(index=["userId"], columns=["movieId"],values="rating")
   # 用户-电影评分矩阵0（用0填充nan）
   train_matrix0 = train_matrix.fillna(0)
   ```

2. 为训练集中的每个用户计算打过分的电影列表，形成字典；
   为训练集中每个电影计算给它评过分的用户列表，形成字典

   ```python
   # 每个用户打过分的电影列表
   dict_key_user_value_movies = {}
   for user in train_matrix.index:
       dict_key_user_value_movies[user]=train_matrix.loc[[user],:].dropna(axis=1).columns
   # 每个电影评过分的用户列表
   dict_key_movie_value_users = {}
   for movie in train_matrix.columns:
       dict_key_movie_value_users[movie]=train_matrix.loc[:,movie].dropna().index
   ```

3. 根据训练数据构造用户相似度矩阵

   ```python
   from sklearn.metrics.pairwise import pairwise_distances
   # 用户相似度矩阵
   user_similarity = pairwise_distances(train_matrix.fillna(0), metric='cosine')
   ```

4. 算法所需的三个辅助函数

   ```python
   # 若待预测电影在训练集没出现，那就没公共打分用户，所以返回提示码999
   def func(x,c_movie):
       l = list(set(dict_key_movie_value_users[x['movieId']]).intersection(set(dict_key_movie_value_users[c_movie])))
       if l:
           return l
       else:
           return 999
   
   # 保证评分一定在0.5~5.0之间，并且保留一位小数
   def s_round(num):
       return round(max(0.5,min(5.0,num)),1)
   
   # 保证评分是0.5、1.0、1.5、2.0这种步长为0.5的数值
   def f_round(num):
       z = int(num)
       f = num - int(num)
       new_f = f
       if f<=0.25:
           new_f = 0
       elif (f>0.25 and f<0.745):
           new_f = 0.5
       elif f>=0.745:
           new_f = 1.0
       return z + new_f
   ```

5. 算法实现。将原始SlopeOne、原始SlopeOne + 加权二部图对最后一步的电影贡献加权、原始SlopeOne + 用户相似矩阵对一开始的单个电影加权、原始SlopeOne + 加权二部图对最后一步的电影贡献加权 + 预测值调整为0.5-5.0步长为0.5的数值的四种预测结果同时算出，输出到csv文件中pred一列

   ```python
   def predict(c_user, c_movie,train_matrix1,train_matrix,dict_key_user_value_movies,dict_key_movie_value_users,user_similarity):
       #先判断当前电影是否在训练集中出现过
       if c_movie not in train_matrix1.columns:
           t1 = time.time()
           pred_rating = train_matrix1.loc[[c_user],:].mean(axis=1)[c_user].round(1)
           t2 = time.time()
           return pred_rating,pred_rating,pred_rating,f_round(pred_rating) # c_user给其他电影的平均分
       else:
           # 待预测用户打过分的电影集合
           c_user_list = dict_key_user_value_movies[c_user]
           t2 = time.time()
           df = pd.DataFrame(c_user_list)
           ### df 一行表示一部待预测用户打过分的电影，下面是对集合中一部电影的操作，逐行
           # 打过两个分的用户列表，长度为l
           df['l'] = df.apply(lambda x: func(x,c_movie),axis=1)
           df = df[df['l']!=999] # 剔除l为空的行
           # 电影在待预测用户手上得分
           df['dev_m_list'] = df.apply(lambda x: train_matrix[x['movieId']][c_user], axis=1)
           # 打分之差列表，长度为l
           df['m_dev_list'] = df.apply(lambda x :[(train_matrix[c_movie][u] - train_matrix[x['movieId']][u]) for u in x['l']],axis=1)
           # 用户相似矩阵对每个用户给出的打分之差加上权重，长度为l
           df['bio_u_list'] = df.apply(lambda x :[user_similarity[int(c_user)-1][u-1] for u in x['l']],axis=1)
           # 二部图求每个电影的权重：\frac{r_{u i} r_{u j}}{\operatorname{deg}(j) \operatorname{deg}(u)}
           df['bio_list'] = df.apply(lambda x :[(train_matrix[c_movie][u]*train_matrix[x['movieId']][u])/(len(dict_key_movie_value_users[x['movieId']])*len(dict_key_user_value_movies[u])) for u in x['l']],axis=1)
           # 当前电影对待预测电影的贡献程度：\sum_{u \in U} \frac{r_{u i} r_{u j}}{\operatorname{deg}(j) \operatorname{deg}(u)}
           df['bio'] = df.apply(lambda x: sum(x['bio_list']), axis=1)
           # 使用用户相似矩阵对打过两个分的用户们加权后得到的一部电影为预测提供的评分差
           df['dev_list'] = df.apply(lambda x: sum(np.multiply(np.array(x['m_dev_list']),np.array(x['bio_u_list'])))/sum(x['bio_u_list']), axis=1)
           # 不使用用户相似矩阵得到的一部电影为预测提供的评分差
           df['dev_old_list'] = df.apply(lambda x: (sum(x['m_dev_list'])/len(x['m_dev_list'])), axis=1)
           #最原始
           pred_rating0 = ((df['dev_old_list'] + df['dev_m_list']).mean())
           #+加权二部图对最后一步的电影贡献加权
           pred_rating1 = (((df['dev_old_list'] + df['dev_m_list'])*df['bio']).sum()/df['bio'].sum())
           #+用户相似矩阵对一开始的单个电影加权
           pred_rating2 = (((df['dev_list'] + df['dev_m_list'])*df['bio']).sum()/df['bio'].sum())
           #+对1取整和0.5
           pred_rating3 = f_round(pred_rating1)
           t3 = time.time()
           return s_round(pred_rating0),s_round(pred_rating1),s_round(pred_rating2),s_round(pred_rating3)
   ```

6. 多进程计算

   ```python
   # 将数据集按行分为8大块，进行并行计算
   def parallelize_dataframe(data, func):
       partitions = 8
       data_split = np.array_split(data, partitions) #划分数据
       pool = multiprocessing.Pool(processes=partitions) 
       data = pd.concat(pool.map(func, data_split)) #处理每个数据块，然后合到一起
       pool.close()
       pool.join()
       return data
   
   def work(data):
       data['pred'] = data.apply(lambda x: predict(x['userId'],x['movieId'],train_matrix,train_matrix0,dict_key_user_value_movies,dict_key_movie_value_users,user_similarity), axis=1)
       return data
   ```

7. 预测所有评分，将pred列拆成pred0~pred3共4列，并保存预测结果

   ```python
   %%time 
   # 预测所有评分
   test = parallelize_dataframe(test, work)
   # 计算MAE得分，并拆分pred列
   for i in range(4):
       test[f'pred{i}'] = test.apply(lambda x: x['pred'][i], axis=1)
       print(round(mean_absolute_error(test[f'pred{i}'], test['rating']),5))
   # 保存
   test.to_csv("../result/all_pred_testing_ratings.csv",index=None,float_format = '%.1f')
   ```



### 三  结果分析和总结

#### 1  结果分析

| 算法及修改                                                   | MAE         | csv中的列名 |
| ------------------------------------------------------------ | ----------- | ----------- |
| 原始SlopeOne                                                 | 0.67117     | pred0       |
| 原始SlopeOne + 加权二部图对最后一步的电影贡献加权            | 0.64963     | pred1       |
| 原始SlopeOne + 加权二部图 + 用户相似矩阵对一开始的单个电影加权 | 0.64983     | pred2       |
| 原始SlopeOne + 加权二部图 + 预测值调整为0.5-5.0步长为0.5的数值 | **0.63841** | pred3       |

预测效果最好的是原始SlopeOne + 加权二部图对最后一步的电影贡献加权 + 预测值调整为0.5-5.0步长为0.5的数值的方法，MAE=0.63841。出人意料的是，在原始SlopeOne+ 加权二部图基础上，加上用户相似矩阵对一开始的单个电影加权，几乎没有效果，经过观察，发现权重系数都较小，我们觉得可能因为这是早期优化，作用被稀疏了。

#### 2  预测结果残差分析

下图是预测评分的残差结果，横坐标是预测值。其中，slope方法因为还加了调整步长，所以我们着重看不调整步长时候的残差图。因为预测值和真实值都是[0,5]，残差是预测值-真实值，所以预测值对应的残差值的分布范围如图中虚线绘制的平行四边形所示，略有超出是因为我们绘制散点图时对数据做了抖动。

如果在1.5~4之间的各个预测值上，残差=0的上下分布数量基本一致，形状也是在y=0上下正态分布，那就说明效果较好。从下图中的第一幅子图中我们可以明显看出，1.5~4之间的各个预测值上残差分布均匀，效果较好。

![Residuals_Predicted](/Residuals_Predicted.png)

下图也是预测评分的残差结果，不过横坐标是真实值，其他说明和上图一致。可以看出，真实值为2.5的电影，样本残差更多是正数，说明预测不准是因为预测大了，如预测成了4。平行四边形中，点的聚集，随着真实值的增大而在与平行四边形的长边保持平行的同时，在往上浮，举例来说，真实值为1的样本的残差不是围绕0分布，而是围绕2分布；真实值为4的样本的残差大致是围绕-0.5分布，更接近0。

虽然真实评分较小时，残差容易是正数，真实评分较大时，残差容易是负数，但是真实评分较小时的残差偏离程度比较大时要大，偏离的更厉害。也就是说，**我们这些算法对真实评分较小的样本预测可能需要调整，要预测的更小些**，因为我们容易把真实值为1的电影打成值为3，说明可能学习的还不够。

![Residuals_True](/Residuals_True.png)

####3  总结和后续改进

##### （1）总结

- 这些方法宝贵的是思想、思路，要达到真正好的效果，还需要结合多种方法的思路，组合到一起。

##### （2）后续改进

- 减少算法时间
- 二部图部分换成用物品相似度矩阵来计算贡献度好像得分没太提高，原因
- 原始SlopeOne + 加权二部图 + 用户相似矩阵对一开始的单个电影加权 变为 原始SlopeOne + 用户相似矩阵对一开始的单个电影加权，控制变量，看看早期用用户相似究竟效果如何
- 算法对真实评分较小的样本预测可能需要调整，要预测的更小些，学习的还不够，例如我们容易把真实值为1的电影打成值为3




### 参考文献

[1] 王冉, 徐怡, 胡善忠, 等. 基于加权二部图的 Slope One 推荐算法[J]. 微电子学与计算机, 2018, 35(3): 93-98.