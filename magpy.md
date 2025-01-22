## 主流模型优化技术

### 基于分析

​	Janus[20], torch.jit.script[2],and TorchDynamo[1]

#### 	torchScript介绍

​	[TorchScript — PyTorch 2.5 文档](https://pytorch.org/docs/stable/jit.html)	 

​	TorchScript 是一种从 PyTorch 代码创建可序列化和可优化模型的方法。 任何 TorchScript 程序都可以从 Python 保存 进程中加载，TorchScript 是 Python 的静态类型子集，因此许多 Python 功能都适用 直接发送到 TorchScript。并在没有 Python 依赖项的进程中加载。

​	 provide tools to incrementally transition a model from a pure Python program to a TorchScript program that can be run independently from Python, such as in a standalone C++ program. This makes it possible to train models in PyTorch using familiar tools in Python and then export the model via TorchScript to a production environment where Python programs may be disadvantageous for performance and multi-threading reasons.

#### 	在 C++ 中加载 TorchScript 模型

​	[使用 C++ 语言加载 TorchScript 模型 — PyTorch 教程 2.5.0+cu124 文档](https://pytorch.org/tutorials/advanced/cpp_export.html)

​	总结：Python 模型到序列化表示形式，可以完全从 C++ *加载*和*执行*，没有 对 Python 的依赖

​	There exist two ways of converting a PyTorch model to Torch Script.

​	The first is known as ***tracing***, a mechanism in which the structure of the model is captured by evaluating it once using example inputs, and recording the flow of those inputs through the model. This is suitable for models that make limited use of control flow. 

​	The second approach is to **add explicit annotations to your model** that inform the Torch Script compiler that it may directly parse and compile your model code, subject to the constraints imposed by the Torch Script language.

***当程序使用编译器尚未支持的灵活的Python特性时，如高级内置函数和复杂的继承，基于分析的方法无法生成图形。因此，这些方法只能从用户程序中提取多个小图片段。通过在缓慢的Python解释器中执行不支持的部分以及在Python和操作符图执行之间频繁的切换，会引入显著的运行时开销。***

### 基于轨迹

Trace-based approaches,
 AutoGraph  ，torch.jit.trace ， torch.fx.symbolic_trace ， Terra  ,LazyTen sor，and Torchy,

generate operator graphs by executing user programs and recording all tensor operations via overloading the tensor operations.

***基于踪迹的方法只能收集一个执行的操作符图，但它们不能知道该图是否会因另一个执行而改变。***

基于追踪获得相关信息，对于图像处理，某次输入的是固定的，可以获得【长，宽，高，通道数】，这个模型可以进行编译器优化

若一个模型中间的输入是不固定的，比如他每一次的输入都在变化，或者中间执行需要上一步的参数是不固定的，就不可以用编译器进行优化。

### 主流方法缺点

**优化需要手动调整，比如图替换和算子融合，难度大，采用编译器优化遇到复杂模型会出现负优化的情况**

![image-20250102144012290](C:\Users\HBY\AppData\Roaming\Typora\typora-user-images\image-20250102144012290.png)

遇到复杂模型时，这两种方法都不能得到最优的实例化结果。

如图所示，与使用同一深度学习编译器直接编译图相比，TorchDynamo平均造成2.08 ×开销( TorchDynamo - Inductor vs . Fullgraph - Inductor)，LazyTensor平均造成2.85 ×开销( LazyTensor-XLA vs . Fullgraph - XLA)。

***Fullgraph进行图替换和算子融合后的情况***

对于静态的模型（输入输出shape固定），可以采用编译器优化，

但是对于动态的图（输入不固定，输出不固定），主流编译器很难进行分析



## MagPy改进

### 理论：

Their operator graphs can be obtained by monitoring tensor operations during program execution.

***Thus,instead of analyzing“what the graph is”like analysis-based approaches, which require a full simulation of a given user code,we only need to know“when the graph will change”.*** 

***Second,only external values can affect program behav ior.***

***Third,both the guard and mock can be determined by ana lyzing program execution states.***



## pytorch-jit-paritybench

ParityBench3is a benchmark that crawls deep learning programs written in PyTorch and with over 100 stars from Github.83%of that benchmark’s 1421 deep learning programs satisfy limited dy namics