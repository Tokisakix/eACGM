import time

from eacgm.sampler import NVMLSampler

sampler = NVMLSampler()

sampler.run()

while True:
    try:
        for sample in sampler.sample(time_stamp=1):
            print(sample)
        time.sleep(2)
        print("---")
    except KeyboardInterrupt:
        break

sampler.close()