# NPU Occupancy Web

一个可直接部署到 GitHub Pages 的静态网页，用于管理 NPU 机器的 24 小时占用时间段。

## 功能

1. 机器信息录入（名称、型号、位置、备注）
2. 24 小时占用看板（按机器、按日期查看）
3. 用户预约时间段（输入姓名、联系方式、占用目的）
4. 冲突校验（时间段重叠会提示）

## 本地运行

直接双击 `index.html` 即可，或使用任意静态服务器打开。

## 部署到 GitHub Pages

### 方式一：仓库根目录部署

1. 把 `npu-occupancy-web` 目录中的文件放到仓库根目录（或设置 Pages 根目录为该目录）。
2. 在 GitHub 仓库中进入 `Settings` -> `Pages`。
3. `Build and deployment` 选择 `Deploy from a branch`。
4. 分支选择 `main`，目录选择 `/ (root)` 或 `/npu-occupancy-web` 对应的发布目录。
5. 保存后等待部署完成，访问 GitHub 提供的网址。

### 方式二：使用 `docs` 目录

1. 把本目录内容复制到仓库 `docs/`。
2. GitHub Pages 目录选择 `/docs`。

## 数据说明

网页数据保存在浏览器 `localStorage`，仅当前浏览器可见，不会自动同步到服务器。
