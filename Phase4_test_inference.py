
import sys
import os

sys.path.append(os.path.abspath("src"))

from cyberbullying.inference.inference_service import predict_post


print("\nTest 1:")
print(predict_post("I am feeling very sad today"))

print("\nTest 2:")
print(predict_post("You are completely useless"))