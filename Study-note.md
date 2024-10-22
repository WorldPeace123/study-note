# 3D

## C/C++关键字之restrict

在C语言中，restrict关键字用于修饰指针(C99标准)。通过加上restrict关键字，编程者可提示编译器：在该指针的生命周期内，其指向的对象不会被别的指针所引用。

需要注意的是，在C++中，并无明确统一的标准支持restrict关键字。但是很多编译器实现了功能相同的关键字，例如gcc和clang中的__restrict关键字。

那么[restrict关键字](https://zhida.zhihu.com/search?content_id=165847176&content_type=Article&match_order=5&q=restrict关键字&zhida_source=entity)能给程序的实际运行带来哪些好处呢？下面举例说明

```cpp
int add1(int* a, int* b)
{
    *a = 10;
    *b = 12;
    return *a + *b;
}
```

大家猜猜`add1`函数的返回值是多少？是10 + 12 = 22吗？

答案是不一定。在指针a和b的地址不同时，返回22没有问题。但是当指针a与b指向的是同一个int对象时，该对象先被赋值为10，后被赋值为12，因此*a和*b都返回12,因此`add1`函数最终返回24

使用`-O3`优化, `add1`对应的汇编代码如下。可以看到，在计算返回值时，为了得到`*a`的值访问了1次内存，而不管在何种条件下(`a == b` or `a != b`)，`*b`的值都是12。因此聪明的编译器将`*a`的值载入`eax`寄存器后，直接加上立即数12，而无需再访问内存获取`*b`的值。在无法确定指针a和b是否相同的情况下，编译器[只能帮你到这里了](https://zhida.zhihu.com/search?content_id=165847176&content_type=Article&match_order=1&q=只能帮你到这里了&zhida_source=entity).

那么[restrict关键字](https://zhida.zhihu.com/search?content_id=165847176&content_type=Article&match_order=5&q=restrict关键字&zhida_source=entity)能给程序的实际运行带来哪些好处呢？下面举例说明

```cpp
int add1(int* a, int* b)
{
    *a = 10;
    *b = 12;
    return *a + *b;
}
```

大家猜猜`add1`函数的返回值是多少？是10 + 12 = 22吗？

答案是不一定。在指针a和b的地址不同时，返回22没有问题。但是当指针a与b指向的是同一个int对象时，该对象先被赋值为10，后被赋值为12，因此*a和*b都返回12,因此`add1`函数最终返回24

使用`-O3`优化, `add1`对应的汇编代码如下。可以看到，在计算返回值时，为了得到`*a`的值访问了1次内存，而不管在何种条件下(`a == b` or `a != b`)，`*b`的值都是12。因此聪明的编译器将`*a`的值载入`eax`寄存器后，直接加上立即数12，而无需再访问内存获取`*b`的值。在无法确定指针a和b是否相同的情况下，编译器[只能帮你到这里了](https://zhida.zhihu.com/search?content_id=165847176&content_type=Article&match_order=1&q=只能帮你到这里了&zhida_source=entity).

通过无restrict和有restrict两种情况下的[汇编指令](https://zhida.zhihu.com/search?content_id=165847176&content_type=Article&match_order=1&q=汇编指令&zhida_source=entity)可看到，后者比前者少访问一次内存，且少执行一条指令。因此我们预期有restrict的版本能够获得可观的性能提升：

注意使用restrict的时候，编程者必须确保不会出现`pointer aliasing`, 即同一块内存无法通过两个或以上的[指针变量名](https://zhida.zhihu.com/search?content_id=165847176&content_type=Article&match_order=1&q=指针变量名&zhida_source=entity)访问。不满足这个条件而强行指定restrict, 将会出现`undefined behavior`



# cuda中threadIdx、blockIdx、blockDim和gridDim的使用

- **线程块(Block)：**由多个线程组成；各block是并行执行的，block间无法通信，也没有执行顺序。
- **线程格(Grid)：**由多个线程块组成。
- **核函数(Kernel)：**在GPU上执行的函数通常称为核函数;一般通过标识符__global__修饰，调用通过<<<参数1,参数2>>>，用于说明内核函数中的线程数量，以及线程是如何组织的。

画个图直观理解一下，下图是1个线程格，里面包含了27块线程块(蓝色的格子)，每个线程块里面又包含了64个线程(绿色的格子)。线程是最小的单位了，虽然这边我画的还是立方体，但通常是看做一个点![img](3D.assets/v2-f8638235a4047c36053028e0150a46c0_720w.webp)

## threadIdx、blockIdx、blockDim和gridDim

以上图为例子，把线程格和线程块都看作一个三维的矩阵。这里假设线程格是一个`3*3*3`的三维矩阵， 线程块是一个`4*4*4`的三维矩阵。

**gridDim**

`gridDim.x`、`gridDim.y`、`gridDim.z`分别表示**线程格**各个维度的大小，所以有

```text
gridDim.x=3    gridDim.y=3   gridDim.z=3
```

**blockDim**

`blockDim.x`、`blockDim.y`、`blockDim.z`分别表示**线程块**中各个维度的大小，所以有

```text
blockDim.x=4   blockDim.y=4  blockDim.z=4
```

**blockIdx**
blockIdx.x、blockIdx.y、blockIdx.z分别表示**当前线程块所处的线程格的坐标位置**

**threadIdx**
threadIdx.x、threadIdx.y、threadIdx.z分别表示**当前线程所处的线程块的坐标位置**

- 线程格里面总的线程个数N即可通过下面的公式算出

```text
N = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z
```

https://zhuanlan.zhihu.com/p/544864997

bank：为了实现内存高带宽的同时访问，NVIDIA GPU中，shared memory和register都被划分成了可以同时访问的等大小内存块(banks)。因此，内存读写n个地址的行为则可以以b个独立的bank同时操作的方式进行，这样有效带宽就提高到了一个bank的b倍。 然而，如果多个线程请求的内存地址被映射到了同一个bank上，那么这些内存请求就变成了串行的，也就是bank conflict



### **内存布局的优化**

- **global memory**

- - *SM* 访问全局内存的逻辑是，先访问 *L1 cache*，如果cache miss，就访问 *L2 cache*，如果L2 cache 也miss，就直接访问显存(DRAM)
  - *global memory* 具有合并访问的特点，所以一个warp所并行的32个thread或更多的内存访问，需要地址连续才能发挥出最大的带宽
  - *global memory* 具有广播的访问的特点，多个thread访问同一个地址，一次完成，广播给所有访问thread



- **L1 cache**

- - *L1 cache* 同样需要合并访存才能发挥最大带宽
  - *L1 cache* 是用于缓存全局内存和局部内存的缓存，其内存读取细粒度是128 bytes，所以内存对齐的访问地址最好是128 bytes的倍数；且访问的数据最好是一次128 bytes(100%带宽利用率)， 再大也只能逐次次访问，因为L1 cache line是128 bytes



- **L2 cache**

- - 所有SM共享一个*L2 Cache*
  - *L2 cache* 同样需要合并访存才能发挥最大带宽
  - *L2 cache* 是用于缓存全局内存和局部内存的缓存，其内存读取细粒度是32 bytes，所以内存对齐的访问最好是32 bytes的倍数



- **shared memory**

- - *shared memory* 的使用场景是多次利用的，数据量较大的数据(比如矩阵乘法中的矩阵数组)，因为这时候*L1 cache*, *L2 cache*, 比较容易缓存缺失(cache line不够大)，进而直接访问全局内存

  - - 我们不要迷信*shared memory*，在小数据量的时候，*shared memory*会造成kernel的访存负担

  - *shared memory* 通过分bank来提供访问的并发性，所以我们在使用 *shared memory* 时，要尽可能避免 **bank conflict**

  - 因为在物理位置上*shared memory* 和 L1 cache 是在一个位置，所以我一开始以为shared memory 的load操作是和L1 cache一致的, 都是异步操作，一次搬移一个cache line， 但是后面查看PTX指令，发现其实就是一个单独的load指令，A100关于异步读取共享内存的设计倒和我原来想法差不多一致

  - - **所以shared memory相较于L1 cache优势到底在哪里呢**？以下是我的个人想法

    - - **不会因为cache miss而打断sp中的流水线，cache miss 后面往往接着计算指令，导致性能下降严重**
      - **而对于大数据量的cache miss往往发生在同一个kernel的多个thread里面，所以warp的切换不能掩盖这个延迟**
      - **share memory的加载一般是发生在kernel初始化的时候，这时候load指令和一些初始化的立即数指令是可以流水线并发，并且可以通过warp切换来隐藏延时**



- **constant memory**

- - *constant memory* 具有片上缓存，延迟低，但是带宽不高，具有广播操作(多个thread访问同一个地址，一次完成，广播给所有访问thread)



- **texture memory**

- - *texture memory* 具有片上缓存，延迟低，带宽高,并且cache结构适合二维数据结构，也就是说将只读图像存入纹理内存，是cache友好的
  - 同时**其片上缓存可以用来缓存全局内存**，成为只读缓存，但是属性是只读的，没有写权限; 可以通过`__ldg`来使用，表示从只读缓存中读取该数据，如果缺失就会从全局内存换入到只读缓存
  - 其片上缓存应该是没有广播机制的，所以如果一个warp的thread同时访问一个地址就会导致多次访问，降低性能; 所以这时候应该使用 *constant memory*

- ### **并发操作的访存优化**

- - 在CUDA中，由于存在多个 *SM*，所以计算和计算，计算和访存操作可以并发，通过 `stream` 的指派来并发任务，不同的流之间没有依赖关系，所以可以将访存和计算分派到不同的流中，完成 overlap 的重叠操作; 一般我们会将当前数据的计算和下一次数据的访问并发起来

- ![img](3D.assets/v2-1e80d32396f0b87b1e9a8ea5bacacabe_1440w.webp)

- 

- - 在我们日常编写算法的时候，经常会遇到累加的情况，比如一个卷积后的数据就需要累加，如果遇到这种情况，单独一个thread进行加法是不划算的，因为这时候其他的thread就会处于停滞的状态，**降低了并发度，没能充分利用内存带宽和指令带宽**

  - - 所以我们可以通过 **reduction** (归约)来解决问题，就拿累加举例，thread1 可以先与 thread2 的值进行累加，thread3 与 thread4 累加......之后 thread1 累加的值再与 thread3 累加的值进行累加......这样就可以提高并发度
    - 一般我们都是使用 *shared memory* 作为中间值的缓存，但是Kepler之后的GPU有提供warp内线程互相访问其寄存器的指令，使用这样的指令就可以降低访问延迟

- ![img](3D.assets/v2-a05f8cb387b8130294f8e426774700bc_1440w.webp)

- 寄存器归约

- ------

- ### **指令的访存优化**

- - 可以通过使用向量 *load* 指令来一次读取多个数据，`float4 float2 int4` 等数据类型的使用，nvcc编译器会在指令选择的时候选择向量 *load* 指令

- ------

- **CPU与GPU的内存交互**

- - GPU作为PCIE设备，访问内存是通过DMA模块来完成，也就是访问的是物理地址。所以 *cudaMemcpy* 只能访问锁定内存，也就是不能被MMU进行换页的内存

  - - 如果该地址是可换页内存，cuda driver会先申请一块临时的锁定内存，然后先copy到临时锁定内存，再从临时锁定内存copy到当前地址，所以这样访存会消耗很长的周期，如果直接将host端的内存分配为锁定内存，这样就会省下不少cycles。但是过多的锁定不可换页内存，会对CPU的多线程造成负担，这需要使用者去权衡利弊

### 优化参考链接

https://www.cnblogs.com/megengine/p/15272175.html





# git

## 配置

```shell
git config --global user.name "WorldPeace123"

git config --global user.email "2510955634@qq.com"

git config --list //查看设置的 信息

ssh-keygen -t rsa -C "2510955634@qq.com"

#*~/.ssh下会有id_rsa.pub公钥，添加到GitHub上即可

ssh git@github.com //验证
```



## 使用

```shell
GitHub上创建新的分支 3d-MPI
git init
git add README.md
git commit -m "first commit" //本地仓库，必须先commit 好本地才能提交远程
git branch -M main
git remote add origin https://github.com/WorldPeace123/3d-MPI.git
git push -u origin main   //推送到远程
```



## 解决 fatal: unable to access xxx: Encountered end of

```shell
git config --global --unset http.proxy 
git config --global --unset https.proxy
```



## git版本控制

```
1.使用 git log 或者 git reflog 命令 获取到要回退或者切换的版本id 

2.使用 git reset --hard [索引值] : 可切换到任意版本[推荐使用这个方式]；

3.git reset --hard 命令会重置  本地仓库、暂存区和工作区，三者的状态保持一致！
```

版本回退/切换的命令：

```doc
1.git reset --hard [索引值] : 可切换到任意版本[推荐使用这个方式]
2.git reset --hard HEAD^ ： 只能后退，一个 ^ 表示回退一个版本，两个^ 表示回退两个版本，。。。依次类推
3.git reset --hard HEAD~n ：只能后退，n表示后退n个版本

^ : 一个^ 表示回退一个版本；
    两个^表示回退两个版本；
    三个^表示回退三个版本；
	n个^表示回退n个版本

git reset --soft  ： 1.仅在本地版本库移动指针。
git reset --mixed : 1.移动本地版本库的指针；2.重置暂存区。（默认的参数）
git reset --hard  :  1.移动本地版本库的指针；2.重置暂存区；3.重置工作区。
```



```shell
[ghfund4_c17@login07 3d-mpi]$ git reflog
795d0ca HEAD@{0}: reset: moving to HEAD^
38481ea HEAD@{1}: commit: add readme.md
795d0ca HEAD@{2}: commit: 3d-mpi
7d0436f HEAD@{3}: commit (initial): mpi 3D
[ghfund4_c17@login07 3d-mpi]$ git reset --hard 38481ea
HEAD is now at 38481ea add readme.md
```

