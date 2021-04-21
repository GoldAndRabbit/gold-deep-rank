> ## gold-deep-rank 
* A well-organized experimental code for Ads/Recsys Ranking process implemented by Tensorflow, adopting tf.estimator api.
* Support for flexible parameter customization, suitble for industrial development.
* Tensorflow version compatibility: support tf 1.14 and tf 2.4.1.

> ## Why Deep CTR model? 
* **Auto Feature Interaction**. 深度学习用于CTR预估问题, 主要优势是通过网络设计达到自动学习特征交互Feature Interaction的目的. 本文中涉及到的模型均是解决Feature Interaction的不同网络设计.
* **Better Sparse ids presentation Support**. 相比GBDT模型, DNN对稀疏id类特征有更好的表示学习能力. 业务需求中往往存在海量且稀疏id类特征, 通过embedding支持对海量id类特征具备较强的表示学习能力.
* **Memorization & Generalization**. 记忆性和泛化性是推荐系统重要的两类能力, 这两类目标通过Wide & Deep Learning结构同时学得, wide part采用FTRL实现, 目的是使得对id类特征具有memorization(记忆性); DNN结构具有generalization的特性(泛化性); 
* 整理实现. 封装在gold-deep-rank这个项目中, repo地址: https://github.com/GoldAndRabbit/gold-deep-rank 主要参考作者源码以及开源库.

> ## Deep CTR framework 
<div align="center">
<img alt="" src="https://z3.ax1x.com/2021/04/10/cdTUkF.png" />
</div>

> ## Wide & Deep Learning
<div align="center">
<img alt="" src="https://s3.ax1x.com/2020/12/15/rMbxQU.png" />
</div>

* Google提出将线性层和DNN同时优化的一般结构, 在此基础上对DNN部分做优化/定制.  
* 泛化性和记忆性是推荐系统的重要的两类基础能力.

> ## DeepFM
<div align="center">
<img alt="" src="https://s3.ax1x.com/2020/12/15/rMqKwd.png" width="400"/>
</div>

<center>   

![deepfm](http://latex.codecogs.com/png.latex?y_{fm}=w_{0}+\sum_{i=1}^{n}w_{i}x_{i}+\sum_{i=1}^{n}\sum_{j=i+1}^{n}<v_{i}\odot{v_{j}}>x_{i}{x_{j}})
</center>

* fm是二阶特征交互的基础方法, 可作为一般Baseline. 
* fm复杂度降低实现推导，将复杂度从O(kn^2)降低到O(kn), 简单记忆方法: sum_square-square_sum, 不要忘了前面还有1/2常系数.


<div align="center">
<img src="https://s3.ax1x.com/2020/12/28/r7PVKS.png" width="400" />
</div>

> ## PNN: Inner/Outer product
<div align="center">
<img alt="" src="https://s3.ax1x.com/2020/12/15/rMqJl8.png"  width="400"/>
</div>

* 向量的内积和外积可以定义两种vec的交叉方式, 很朴素的feature interaction思想.

> ## DCN: Cross Network
<div align="center">
<img alt="" src="https://s3.ax1x.com/2020/12/15/rMbzyF.png"  width="400"/>
</div>

* 思想是实现**多项式形式**的feature interaction，其实和一般意义上的特征交叉有所区别.
<center>

![dcn](http://latex.codecogs.com/png.latex?x_{l+1}=x_{0}x_{l}^{T}x_{l}+b_{l}+x_{l})
</center>

> ## xDeepFM: CIN
<div align="center">
<img alt="" src="https://s3.ax1x.com/2020/12/15/rMqFF1.png" />
</div>

* 引入vector-wise feature交叉, 而不是bit-wise.
* CIN的结构不建议理解公式(形式化复杂), 结合图和源码看比较容易理解.


> ## AFM: FM based attention
<div align="center">
<img alt="" src="https://s3.ax1x.com/2020/12/15/rMqPoR.png"  width="666"/>
</div>

* 在原有deepfm基础上, 加一层attention layer
<center>

![afm](http://latex.codecogs.com/png.latex?y_{fm}^=w_{0}+\sum_{i=1}^{n}w_{i}x_{i}+\mathbf{p^{T}}\alpha_{ij}\sum_{i=1}^{n}\sum_{j=i+1}^{n}<v_{i}\odot{v_{j}}>x_{i}{x_{j}})
</center>

> ## AutoInt: Multi-head attention
<div align="center">
<img alt="" src="https://s3.ax1x.com/2020/12/15/rMxVSI.png"  width="400"/>
</div>

* 引入multi-head self attention学习feature interaction, 关于multi-head self attention查看transformer原理


> ## FiBiNet: SENET & Bi-linear interaction
<div align="center">
<img alt="" src="https://s3.ax1x.com/2020/12/15/rMqY6S.png"  width="666"/>
</div>

* 引入SENET学习feature interaction

> ## Dataset Description
|Dataset|Description|
|----|----|
|Census Incomes|Extraction was done by Barry Becker from the 1994 Census database. Prediction task is to determine whether a person makes over 50K a year.|
|Alibaba Display Ads|Alibaba Display ADs|
|Criteo|To be updated...|
|Avazu|To be updated...|
|Tencent Social Ads|To be updated...|

> ## Evaluation
To be updated...

> ## Update Log
* 20210421: Support tf 2.4.1 version.
* 20210410: Fix census csv data read bug. Update README.md: add deep interaction docs.

