# 运行启动说明

框架: Django 5.1 其余依赖均是系统库

torch: 2.1.1+cu118 及相关生态

opencv: 4.9.0.80

进入 Handwritten_digit_recognition 目录下，若 python 已经配置在环境变量中，可在命令行中直接输入：

```shell
python manage.py runserver 127.0.0.1:8000
```

即可在本机的 8000 端口上上启动服务。若使用别的端口修改 :8000 即可。

还可以使用 Pycharm 打开 Handwritten_digit_recognition 目录，即可打开项目。在 Pycharm 终端中输入上述语句即可在本机的 8000 端口上上启动服务。
