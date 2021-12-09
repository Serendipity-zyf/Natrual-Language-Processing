# NLP开发环境的搭建

## 1.1 为什么使用Python来开发NLP？
* Python有丰富的自然语言处理库：jieba、ntlk、spacy...
* 语法简单，入门快
* 有很多数据科学相关的库：numpy、pandas、matplotlib......

## 1.2 Anaconda的下载与使用
* Windows、Linux、Macos（基于intel系列）可以直接从 **Anaconda** 官方网址进行下载：[https://www.anaconda.com/](https://www.anaconda.com/)
* Macos（基于Apple silicon M1系列）可以按照这个回答进行操作：[https://blog.csdn.net/weixin_47614014/article/details/118070452?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3.no_search_link](https://blog.csdn.net/weixin_47614014/article/details/118070452?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3.no_search_link)

## 1.3 conda创建虚拟环境
（Macos、Linux系统）参考1:[https://blog.csdn.net/weixin_47614014/article/details/118070452?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3.no_search_link](https://blog.csdn.net/weixin_47614014/article/details/118070452?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3.no_search_link)

（Windows系统）参考2:[https://www.bilibili.com/video/BV1jV411x7JQ?from=search&seid=17877063708177251341&spm_id_from=333.337.0.0](https://www.bilibili.com/video/BV1jV411x7JQ?from=search&seid=17877063708177251341&spm_id_from=333.337.0.0)

## 1.4 jupyter notebook的概念以及优点
**jupyter notebook 是已经成为一个几乎支持所有语言，能够把软件代码、计算输出、解释文档、多媒体资源整合在一起的多功能科学运算平台。**

**1、整合所有的资源**

在真正的软件开发中，上下文切换占用了大量的时间。什么意思呢？举个例子你就很好理解了，比如你需要切换窗口去看一些文档，再切换窗口去用另一个工具画图等等。这些都是影响生产效率的因素。 jupyter notebook 通过把所有和软件编写有关的资源全部放在一个地方，解决了这个问题。当你打开一个 jupyter notebook 时，就已经可以看到相应的文档、图表、视频和相应的代码。这样，你就不需要切换窗口去找资料，只要看一个文件，就可以获得项目的所有信息。

**2、交互性编程体验**

在机器学习和数学统计领域，Python 编程的实验性特别强，经常出现的情况是，一小块代码需要重写 100 遍，比如为了尝试 100 种不同的方法，但别的代码都不想动。这一点和传统的 Python 开发有很大不同。如果是在传统的 Python 开发流程中，每一次实验都要把所有代码重新跑一遍，会花费开发者很多时间。特别是在像 Facebook 这样千万行级别的代码库里，即使整个公司的底层架构已经足够优化，真要重新跑一遍，也需要几分钟的时间。而 jupyter notebook 引进了 Cell 的概念，每次实验可以只跑一小个 Cell 里的代码；并且，所见即所得，在代码下面立刻就可以看到结果。这样强的互动性，让 Python 研究员可以专注于问题本身，不被繁杂的工具链所累，不用在命令行直接切换，所有科研工作都能在 jupyter notebook上完成。

## 1.5 jupyter notebook的使用

* Windows、Linux、Macos（基于intel系列）可以直接从下载 Anaconda 中打开
* Macos（基于Apple silicon M1系列）可以在终端中输入 conda install jupyter notebook ，下载完成之后再次输入在终端中输入jupyter notebook以启动

## 1.6 关于jupyter notebook如何切换虚拟环境（进阶操作）
参考：[https://www.bilibili.com/video/BV1jV411x7JQ?from=search&seid=17877063708177251341&spm_id_from=333.337.0.0](https://www.bilibili.com/video/BV1jV411x7JQ?from=search&seid=17877063708177251341&spm_id_from=333.337.0.0)
