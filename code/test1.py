import time

bg = time.time()
time.sleep(0.1)
ed = time.time()
print(bg)
print(f"take {ed - bg} seconds to train")