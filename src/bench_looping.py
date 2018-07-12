import numpy as np
from looping import chunked_loop, chunked_loop_b

size_M = 20
N = size_M * 250000

chunk_sizes = np.logspace(2, 23, base=2, num=250, dtype=int)

with open('bench_chunked_loop.csv', 'w') as f:
    f.write('chunk_size,time1,time2\n')
    for i, chunk_size in enumerate(chunk_sizes):
        X = np.zeros(N).astype(np.float32)
        t1 = chunked_loop(X, chunk_size)
        t2 = chunked_loop_b(X, chunk_size)
        s = str(chunk_size) + ',' + str(t1) + ',' + str(t2) + '\n'
        f.write(s)
        print(str(i/len(chunk_sizes)*100)+'%')
