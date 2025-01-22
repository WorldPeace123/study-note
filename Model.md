# 关于模型优化做的工作

## 完成DCU移植

libtorch提供C++接口，性能比Pytho版本提升0.1（有总比没有强）

在此切换Torch_DIR

```shell
export LD_LIBRARY_PATH="/home/worldpeace/soft/libtorch/lib:$LD_LIBRARY_PATH"
export PATH="/home/worldpeace/soft/deepmd-c++/bin:$PATH"
export LD_LIBRARY_PATH="/home/worldpeace/soft/deepmd-c++/lib:$LD_LIBRARY_PATH"
cmake -L -C ../cmake/presets/basic.cmake  \ 
-C ../cmake/presets/kokkos-openmp.cmake \    
-C ../cmake/presets/kokkos-cuda.cmake \      
-DCMAKE_BUILD_TYPE=Release \      
#   取消注释二选一    #
#原始版本# -DTorch_DIR=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`/Torch \
#libtorch版本# -DTorch_DIR=/home/worldpeace/soft/libtorch/share/cmake/Torch \   
-DGFLAGS_INCLUDE_DIR=/home/worldpeace/soft/libtorch/include \      
-DCUDA_ARCH=AMPERE86 \      
-DMKL_INCLUDE_DIR=/opt/intel/oneapi/mkl/latest/include \      
-DCMAKE_PREFIX_PATH=/home/worldpeace/soft/deepmd-kit\  
-DCMAKE_INSTALL_PREFIX=/opt/LMP_dp_allegro_C -DBUILD_TOOLS=ON -DBUILD_SHARED_LIBS=ON \  
-DPKG_GPU=ON  \        
-DFFT=FFTW3 -DFFTW3_LIBRARY=/opt/fftw3/lib/libfftw3.so \   
-DFFTW3_INCLUDE_DIR=/opt/fftw3/include \    
-DLAMMPS_INSTALL_RPATH=ON  ../cmake
```

TVM调研实践，卡在Input

## input代码分析，获得Input形状

/home/worldpeace/anaconda3/envs/tvm/lib/python3.11/site-packages/nequip/ase/nequip_calculator.py

```python
     def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Calculate properties.

        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        data = AtomicData.from_ase(atoms=atoms, r_max=self.r_max)
        for k in AtomicDataDict.ALL_ENERGY_KEYS:
            if k in data:
                del data[k]
        data = self.transform(data)
        data = data.to(self.device)
        data = AtomicData.to_AtomicDataDict(data)

        # predict + extract data
        out = self.model(data)
```

这里data为模型需要的输入，获得并保存在txt内部

```
{'edge_index': tensor([[ 0,  0,  0,  ..., 70, 70, 70],        [ 3,  5,  6,  ..., 47, 42, 18]], device='cuda:0'), 'pos': tensor([[1.1242e-01, 1.2245e+01, 4.1290e+00],        [3.6315e-01, 4.2438e+00, 8.0531e+00],        [9.3500e-02, 4.5374e+00, 4.3283e+00],        [1.2334e-01, 8.6023e+00, 8.1177e+00],        [1.2322e+01, 8.6604e+00, 4.3235e+00],        [4.2572e+00, 1.2474e+01, 8.0820e+00],        [4.3863e+00, 1.2205e+01, 4.0424e+00],        [4.3404e+00, 3.7507e+00, 8.1078e+00],        [4.3428e+00, 3.9973e+00, 3.9187e+00],
```

添加tensordict库，修改格式为可识别

```
input=AtomicData.to_AtomicDataDict({'edge_index': tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,          1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,          1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,          1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,          2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,          2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
```

jit.load script model。然后运行

```python
import torch
from torch import tensor
from nequip.data import AtomicData, AtomicDataDict
from tensordict.tensordict import TensorDict
mod=torch.jit.load('deployed.pth')
mod=mod.to('cuda')
input=AtomicData.to_AtomicDataDict({'edge_index': tensor([[ 
out=mod(input)
print(out)
```

正确性对比：

上面两个python形式已经对比无误

nequip_calculator.py out 和LAMMPS pair_allegro.cpp的out对比即可



ASE仿真，完成模型独立运行

## Script model保存机制，C++调用单步性能插桩

pair_allegro.cpp输出

```
Per MPI rank memory allocation (min/avg/max) = 5.31 | 5.31 | 5.31 Mbytes   Step          Time          PotEng         KinEng         TotEng          Temp          Press          Volume        Density       
0   0             -5115.3514      247.66244     -4867.6889      1000           124912.37      35001.599      9.8101743  
model.forward Time is : 0.445529 stoTensor().cpu Time is : 0.000252 s
Pair All Time is : 0.452624 s
model.forward Time is : 3.49675 s
toTensor().cpu Time is : 7.3e-05 s
Pair All Time is : 3.50579 s
model.forward Time is : 0.283893 s
toTensor().cpu Time is : 0.000103 s
Pair All Time is : 0.286683 s
model.forward Time is : 0.282643 s
toTensor().cpu Time is : 6e-05 s
Pair All Time is : 0.290819 s
model.forward Time is : 0.283949 s
toTensor().cpu Time is : 0.000106 s
Pair All Time is : 0.28688 s
model.forward Time is : 0.282495 s
toTensor().cpu Time is : 5.4e-05 s
Pair All Time is : 0.286974 s
model.forward Time is : 0.282829 s
toTensor().cpu Time is : 0.000113 s
```

# python 单独load Model对比

```
/home/worldpeace/anaconda3/envs/tvm/lib/python3.11/site-packages/nequip/__init__.py:20: UserWarning: !! PyTorch version 2.5.1 found. Upstream issues in PyTorch versions 1.13.* and 2.* have been seen to cause unusual performance degredations on some CUDA systems that become worse over time; see https://github.com/mir-group/nequip/discussions/311. The best tested PyTorch version to use with CUDA devices is 1.11; while using other versions if you observe this problem, an unexpected lack of this problem, or other strange behavior, please post in the linked GitHub issue.  warnings.warn(
Time :0.6278s
Time :0.5497s
Time :1.6636s
Time :0.3279s
Time :0.0221s
Time :0.0236s
Time :0.0239s
Time :0.0243s
Time :0.0231s
Time :0.0288s
```

## 性能分析，不同硬件平台底层走不同计算库

Z100

![image-20250108102238370](C:\Users\HBY\AppData\Roaming\Typora\typora-user-images\image-20250108102238370.png)

RTX3090

![image-20250108102307083](C:\Users\HBY\AppData\Roaming\Typora\typora-user-images\image-20250108102307083.png)

***性能分析脚本***

找到主函数，替换

显示底层计算库算子耗时

```python
import torch.autograd.profiler as profiler
```



```python
if __name__ == "__main__":
#    cProfile.run('main(running_as_script=True)')

    with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as p:
        main(running_as_script=True)
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
```

torchboard的分析

```python
with torch.profiler.profile(    
activities=[        
torch.profiler.ProfilerActivity.CPU, 
torch.profiler.ProfilerActivity.CUDA,  
],   
on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/torchboard'), 
record_shapes=True,  
profile_memory=True,   
with_stack=True) as p:
	main(running_as_script=True)
```

tensorboard可视化分析 only_run_model

```shell
tensorboard --logdir=log --host=127.0.0.1
```

K100 torchprof

![image-20250113154234771](C:\Users\HBY\AppData\Roaming\Typora\typora-user-images\image-20250113154234771.png)



Magpy调研与应用，失败

## 后续方向

先重新保存模型 不用script model

尝试TVM编译model，Magpy