+++
date = '2026-05-17T16:36:24+01:00'
draft = true
title = 'Fast vLLM: Reducing container cold-start time by 10x'
summary = 'Reducing container start-up time from 81s to 6s via cache re-use and checkpoint-restore in userspace (criu)'
+++

Don't you hate it when vLLM takes an eternity to boot? Me too.

1. Introduction and summary (overview of approach, results)
1. Discuss what happens on vLLM startup - what's compiled (and is it cached?)  (torch.compile, CUDA graph capture, weight downloading, etc.)
2. Discuss a simple approach whereby we simply remount the cache from disk into the container on next startup -- capture once and reuse
3. Introduce `criu`: what it does and how it works (a shallow dive into the interals to give a high-level view of what's happening). How it can be used in our situation.
4. Putting vLLM to sleep (sleep level 1 and sleep level 2), and why this is beneficial for checkpoints. Restoring these checkpoints -- quantify bottlenecks and timings.
5. Optimisations: use a `criu` fork that uses `mmap` instead of `preadv`.
6. Optimisations: use a custom weight loader. How it works, why it's fast, why vLLM's native weightloader is suboptimal. Maybe mention a side note that I tried to use vLLM's kernel weights reloader but that was actually slower.
7. Future directions etc.
8. Problems and quirks encountered, how they were solved.

Use Excalidraw for diagrams.

NVMe floor for weight reloading (3764mb/s; for 6gb weights that's around 1.5s):
```
joel@joels-desktop:~/Projects/j9smith.github.io$ 
sudo hdparm -t --direct /home/joel/Projects/fast-vllm/experiments/3/weights/weights.bin

fio --name=seq_read --filename=/home/joel/Projects/fast-vllm/experiments/3/weights/weights.bin \
    --rw=read --bs=1M --direct=1 --numjobs=1 \
    --size=10G --runtime=10 --time_based=0
[sudo] password for joel: 

/home/joel/Projects/fast-vllm/experiments/3/weights/weights.bin:
 Timing O_DIRECT disk reads: read(2097152) returned 1923072 bytes
BLKFLSBUF failed: Inappropriate ioctl for device
seq_read: (g=0): rw=read, bs=(R) 1024KiB-1024KiB, (W) 1024KiB-1024KiB, (T) 1024KiB-1024KiB, ioengine=psync, iodepth=1
fio-3.36
Starting 1 process
seq_read: Laying out IO file (1 file / 10240MiB)
Jobs: 1 (f=1): [R(1)][-.-%][r=3689MiB/s][r=3689 IOPS][eta 00m:00s]
seq_read: (groupid=0, jobs=1): err= 0: pid=15362: Thu May 28 09:17:12 2026
  read: IOPS=3589, BW=3589MiB/s (3764MB/s)(10.0GiB/2853msec)
    clat (usec): min=242, max=9639, avg=274.92, stdev=98.68
     lat (usec): min=242, max=9639, avg=274.94, stdev=98.68
    clat percentiles (usec):
     |  1.00th=[  249],  5.00th=[  251], 10.00th=[  253], 20.00th=[  260],
     | 30.00th=[  262], 40.00th=[  265], 50.00th=[  269], 60.00th=[  269],
     | 70.00th=[  277], 80.00th=[  289], 90.00th=[  297], 95.00th=[  306],
     | 99.00th=[  396], 99.50th=[  408], 99.90th=[  457], 99.95th=[  506],
     | 99.99th=[ 2606]
   bw (  MiB/s): min= 3334, max= 3714, per=100.00%, avg=3613.60, stdev=158.31, samples=5
   iops        : min= 3334, max= 3714, avg=3613.60, stdev=158.31, samples=5
  lat (usec)   : 250=4.79%, 500=95.16%, 750=0.03%, 1000=0.01%
  lat (msec)   : 4=0.01%, 10=0.01%
  cpu          : usr=0.07%, sys=10.03%, ctx=10260, majf=0, minf=268
  IO depths    : 1=100.0%, 2=0.0%, 4=0.0%, 8=0.0%, 16=0.0%, 32=0.0%, >=64=0.0%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     issued rwts: total=10240,0,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=1

Run status group 0 (all jobs):
   READ: bw=3589MiB/s (3764MB/s), 3589MiB/s-3589MiB/s (3764MB/s-3764MB/s), io=10.0GiB (10.7GB), run=2853-2853msec

Disk stats (read/write):
  nvme0n1: ios=81394/69, sectors=20807440/3440, merge=0/0, ticks=14999/12, in_queue=15012, util=88.57%
```

<iframe
  src="/widgets/triton-explorer/index.html"
  style="width:100%;height:680px;border:none;border-radius:8px;"
  loading="lazy">
</iframe>