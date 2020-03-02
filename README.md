# Particle-Filter
A localization algorithm based on particle filter.  
对源代码尝试如下改造：  
1）基于particles的位置，计算robot的唯一位置并输出到屏幕上  
思路：  
由于particles都是距离robot最近的点，所以考虑直接计算所有particles的x轴y轴坐标的平均值作为robot的坐标。line127-135  
输出则是在循环中加入一行，将robot的坐标放入格式字符串中显示出来。line189-190
2）修改weights的分布为帕累托分布（当前使用的是正态分布）
思路：  

