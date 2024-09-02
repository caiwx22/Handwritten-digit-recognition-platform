from django.db import models


class Task(models.Model):
    task_name = models.CharField(max_length=255, unique=True)
    lr = models.FloatField()
    epoch = models.IntegerField()
    batch_size = models.IntegerField()
    loss_epoch = models.TextField()
    accuracy_epoch = models.TextField()
    start_time = models.DateTimeField()
    end_time = models.DateTimeField()
    duration = models.FloatField()

    def __str__(self):
        return self.task_name
