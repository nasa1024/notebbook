---
title: 保存双色球数据保存为csv
tags:
  - 双色球
  - 数据抓取
createTime: 2024/11/01 16:02:01
permalink: /article/7fsiywqq/
---
将会介绍一直快速的从`500彩票网`中导出双色球数据为csv文件的方法

<!-- more -->

# 第一步
进入 [https://datachart.500.com/ssq/history/history.shtml](https://datachart.500.com/ssq/history/history.shtml)
# 第二步
按下f12,选择***控制台***，粘贴如下代码即可

```
let bom = []
for (let index = 0; index < document.querySelector("#tdata").childElementCount; index++) {
    text = document.querySelector(`#tdata > tr:nth-child(${index + 1})`).innerText
    const pattern = /^(\d+\t){8}/;
    const extracted = text.match(pattern)[0];
    const numbers = extracted.split('\t').map(Number);
    bom.push(numbers)
    console.log('done')
}
// 将数据转换为 CSV 格式
const csvContent = bom.map(row => row.join(',')).join('\n');

// 创建一个包含 CSV 数据的 Blob 对象
const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });

// 生成 Blob 对象的下载链接
const url = URL.createObjectURL(blob);

// 创建一个 <a> 元素用于下载
const link = document.createElement('a');
link.href = url;
link.download = 'data.csv'; // 指定下载的文件名

// 模拟点击下载链接
link.click();

// 释放 Blob 对象的 URL
URL.revokeObjectURL(url);
```
## 提示
输入你想要的期数
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/4caf55cbe3e26f948823e3cc6448a2e6.png)