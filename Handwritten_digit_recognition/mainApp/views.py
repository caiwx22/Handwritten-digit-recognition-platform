from django.shortcuts import render, redirect
from django.http import HttpResponse
from recognitionModel.train import train
from recognitionModel.inference import run_inference
from datetime import datetime
from django.views.decorators.csrf import csrf_exempt
import base64
import os
import glob

from .models import Task  # Import the Task model


def defaultJump(request):
    return redirect("/index/")


def hello(request):
    return render(request, "index.html")


def training(request):
    return render(request, "training.html")


def getParameter(request):
    # 获取表单参数
    task_name = str(request.GET.get('taskName'))
    lr = float(request.GET.get('lr'))
    epoch = int(request.GET.get('epoch'))
    batch_size = int(request.GET.get('batchsize'))

    # 检查任务名称是否唯一，如果不是则重命名
    original_task_name = task_name
    suffix = 1
    while Task.objects.filter(task_name=task_name).exists():
        task_name = f"{original_task_name}({suffix})"
        suffix += 1

    # 训练
    start_time = datetime.now()
    loss_epoch, accuracy_epoch = train(epoch, batch_size, lr, 10, task_name)
    end_time = datetime.now()
    duration = round((end_time - start_time).total_seconds(), 2)

    # 创建并保存新任务
    Task.objects.create(
        task_name=task_name,
        lr=lr,
        epoch=epoch,
        batch_size=batch_size,
        loss_epoch=str(loss_epoch).replace("{", "").replace("}", "").replace("'", "").replace(", ", ",\n"),
        accuracy_epoch=str(accuracy_epoch).replace("{", "").replace("}", "").replace("'", "").replace(", ", ",\n"),
        start_time=start_time,
        end_time=end_time,
        duration=duration
    )

    return redirect('/results/')


def results(request):
    # 从数据库中获取所有任务
    tasks = Task.objects.all().order_by('id')
    context = {'tasks': tasks}
    return render(request, "results.html", context)


def deleteInfo(request):
    # 从请求中获取任务名称
    task_name = str(request.GET.get('taskName'))

    # 删除与任务名称匹配的任务
    try:
        # 从数据库中删除所有与 task_name 匹配的记录
        Task.objects.filter(task_name=task_name).delete()
    except Task.DoesNotExist:
        return HttpResponse("Task not found")

    # 删除相应的模型文件
    files_to_delete = glob.glob(os.path.join("recognitionModel/models", f"{task_name}.pth"))

    for file_path in files_to_delete:
        try:
            os.remove(file_path)  # 删除文件
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    return redirect('/results/')


def useModel(request):
    task_name = str(request.GET.get('taskName'))
    context = {'taskName': task_name}
    return render(request, "useModel.html", context)


def uploadImage(request):
    model = str(request.GET.get('model'))
    uploaded_file = request.FILES.get('image')
    if uploaded_file:
        # 保存到文件系统
        with open('static/target.jpg', 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        label = run_inference(10,
                              'static/target.jpg',
                              f'recognitionModel/models/{model}.pth')
        context = {'label': label}
        return render(request, "showImg.html", context)

    else:
        return HttpResponse("Upload Image Error")


@csrf_exempt
def save_signature(request):
    model = str(request.GET.get('model'))
    img_data = request.POST.get('imgData')
    if img_data:
        # 将base64字符串转换为图片并保存
        img_data = img_data.split(',')[1]  # 去掉前缀data:image/png;base64,
        img_data = base64.b64decode(img_data)
        with open('static/signature.jpeg', 'wb') as f:
            f.write(img_data)

        label = run_inference(10,
                              'static/signature.jpeg',
                              f'recognitionModel/models/{model}.pth')
        context = {'label': label}
        return render(request, "showHw.html", context)

    else:
        return HttpResponse("Written digit save Error")
