# Particle-Filter
A localization algorithm based on particle filter.
对源代码尝试如下改造：
1）基于particles的位置，计算robot的唯一位置并输出到屏幕上
思路：
由于particles都是距离robot最近的一些点，所以考虑直接计算所有particles的x轴y轴坐标的平均值作为robot的坐标。代码在line
输出则是在循环中加入一行
