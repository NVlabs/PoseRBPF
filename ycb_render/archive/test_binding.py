import CppYCBRenderer
import numpy as np
import cv2
r = CppYCBRenderer.CppYCBRenderer(256, 256, 0)
print(r)
r.init()

r.query()

a = np.zeros((256,256,3), dtype=np.float32)
r.draw_py(a)
print(a)
for i in range(1000):
    r.draw(a)
print(a)

r.release()

for _ in range(100):
    cv2.imshow('test', a)
    cv2.waitKey(10)
