# The shape population per table (D-170 §1)

- current criterion — **at least 2 distinct pipeline_kind in the candidate space**
- previous criterion — align_a/b/c == 8 on every candidate (until 2026-09-13)

| table | table shapes | ★ population | previous | dropped |
|---|---:|---:|---:|---|
| a6000 | 66 | **65** | 61 | 1024x4096x4097 |
| 5090 | 66 | **65** | 61 | 1024x4096x4097 |
| 4090 | 64 | **63** | 59 | 1024x4096x4097 |
| h100 | 64 | **63** | 59 | 1024x4096x4097 |

## Every shape either criterion treats differently

| table | verdict | shape | candidates | stages | families | align | vendor regret |
|---|---|---|---:|---|---|---|---:|
| a6000 | dropped | `1024x4096x4097` | 4,800 | 2..2 | pipelined | [1, 1, 8] | 1.1875 |
| a6000 | readmitted ★ | `1024x4096x4098` | 17,250 | 2..8 | multistage+pipelined | [2, 2, 8] | 1.1565 |
| a6000 | readmitted ★ | `1024x4096x4100` | 17,250 | 2..8 | multistage+pipelined | [4, 4, 8] | 1.2031 |
| a6000 | readmitted ★ | `1024x4100x4096` | 16,315 | 2..8 | multistage+pipelined | [8, 8, 4] | 1.0432 |
| a6000 | readmitted ★ | `1024x4098x4096` | 16,315 | 2..8 | multistage+pipelined | [8, 8, 2] | 1.0400 |
| 5090 | dropped | `1024x4096x4097` | 6,080 | 2..2 | pipelined | [1, 1, 8] | 1.4512 |
| 5090 | readmitted ★ | `1024x4096x4100` | 21,945 | 2..8 | multistage+pipelined | [4, 4, 8] | 1.1655 |
| 5090 | readmitted ★ | `1024x4096x4098` | 21,945 | 2..8 | multistage+pipelined | [2, 2, 8] | 1.1485 |
| 5090 | readmitted ★ | `1024x4098x4096` | 21,335 | 2..8 | multistage+pipelined | [8, 8, 2] | 1.2333 |
| 5090 | readmitted ★ | `1024x4100x4096` | 21,505 | 2..8 | multistage+pipelined | [8, 8, 4] | 1.2444 |
| 4090 | dropped | `1024x4096x4097` | 6,080 | 2..2 | pipelined | [1, 1, 8] | 1.1069 |
| 4090 | readmitted ★ | `1024x4096x4100` | 21,850 | 2..8 | multistage+pipelined | [4, 4, 8] | 1.0433 |
| 4090 | readmitted ★ | `1024x4096x4098` | 21,850 | 2..8 | multistage+pipelined | [2, 2, 8] | 1.0299 |
| 4090 | readmitted ★ | `1024x4100x4096` | 21,335 | 2..8 | multistage+pipelined | [8, 8, 4] | 1.0871 |
| 4090 | readmitted ★ | `1024x4098x4096` | 21,335 | 2..8 | multistage+pipelined | [8, 8, 2] | 1.0913 |
| h100 | dropped | `1024x4096x4097` | 6,175 | 2..2 | pipelined | [1, 1, 8] | 1.1363 |
| h100 | readmitted ★ | `1024x4096x4098` | 40,375 | 2..8 | multistage+pipelined | [2, 2, 8] | 1.6202 |
| h100 | readmitted ★ | `1024x4096x4100` | 40,375 | 2..8 | multistage+pipelined | [4, 4, 8] | 1.6941 |
| h100 | readmitted ★ | `1024x4098x4096` | 39,695 | 2..8 | multistage+pipelined | [8, 8, 2] | 1.0653 |
| h100 | readmitted ★ | `1024x4100x4096` | 39,695 | 2..8 | multistage+pipelined | [8, 8, 4] | 1.0701 |

★ `readmitted` are the shapes the alignment criterion excluded and the kernel-family criterion keeps.
