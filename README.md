# Particle-Filter
A localization algorithm based on particle filter.  
源代码地址http://ros-developer.com/2019/04/10/parcticle-filter-explained-with-python-code-from-scratch/  
对源代码尝试如下改造：  
1）基于particles的位置，计算robot的唯一位置并输出到屏幕上  
2）修改weights的分布为帕累托分布（当前使用的是正态分布）  
3）为landmark和robot之间的距离增加随机误差，观察定位结果

