# src/model_training/__init__.py
from .teacher_trainer import TeacherModelTrainer, train_all_teachers
from .student_trainer import StudentModelTrainer, train_xgboost_student