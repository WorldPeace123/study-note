## 问题1：

## 官方给出pytorch和dtk的版本是否会影响性能？

移植的机器学习势基于pytorch,采用光合开发者社区提供的whl安装，dtk用23.10.Python3.8。切换新版本对整体性能是否有较为明显的提升？这方面有无官方测试

## 问题2：

## 对于模型profiling，底层Cijk_Ailk_Bljk_SB_MT128x64x8_SN_APM1_AF0EM1_AF1EM1_A... 能否进一步优化？

我们针对不同硬件平台做了性能分析，发现DCU和NV底层走的计算库不同。而移植所用pytorch是开发者社区提供的Whl,目前我们这边应该是无法针对源码进行操作。而热点函数能否进一步优化？除hipprof外有无其他性能分析工具？

NVIDIA3090

<img src="C:\Users\HBY\AppData\Roaming\Typora\typora-user-images\image-20250107140836360.png" alt="image-20250107140836360" style="zoom: 50%;" />

Z100L

<img src="C:\Users\HBY\AppData\Roaming\Typora\typora-user-images\image-20250107141030022.png" alt="image-20250107141030022" style="zoom: 50%;" />

## 问题3：

## 是否可以有对标Libtorch的DCU版本的库？

<img src="C:\Users\HBY\AppData\Roaming\Typora\typora-user-images\image-20250107142906530.png" alt="image-20250107142906530" style="zoom:50%;" />