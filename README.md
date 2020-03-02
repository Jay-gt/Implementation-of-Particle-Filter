# Particle-Filter
A localization algorithm based on particle filter.  
对源代码尝试如下改造：  
1）基于particles的位置，计算robot的唯一位置并输出到屏幕上  
2）修改weights的分布为帕累托分布（当前使用的是正态分布）  
3）为landmark和robot之间的距离增加随机误差，观察定位结果

![Alt text](https://github.com/Jay-gt/Particle-Filter/blob/master/%E6%94%B9%E9%80%A01%E6%88%AA%E5%9B%BE.jpg)
