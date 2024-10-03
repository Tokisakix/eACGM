import time

from eacgm.sampler import GPUSampler

sampler = GPUSampler()

sampler.run()

while True:
    try:
        for sample in sampler.sample():
            print(sample)
        time.sleep(1)
        print("---")
    except KeyboardInterrupt:
        break

sampler.close()