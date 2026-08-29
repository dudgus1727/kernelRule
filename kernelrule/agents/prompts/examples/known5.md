## 형태 예시 — ★ **이미 라이브러리에 있는 것들입니다**

아래 셋은 위 "이미 있는 피처" 에 들어 있습니다. **다시 만들지 마세요** —
중복으로 거부되어 예산만 씁니다.

물리를 어떻게 코드로 옮기는지, 출처를 어떻게 적는지, 그리고 **형태가
어떻게 다른지**만 보세요.

```python
# (1) 비율형 — 0~1 로 정규화되고 물리적 상한이 있다
def tail_waste(p, hw, cfg) -> float:
    """마지막 wave 에서 노는 SM 슬롯의 비율. 클수록 나쁘다.

    CTA 가 SM 에 나뉘어 배분되는데, 마지막 묶음에서 일부 SM 이 논다.
    출처: CUDA C++ Best Practices Guide, "Thread and Block Heuristics"
    """
    gm = math.ceil(p.M / cfg.tile_m)
    gn = math.ceil(p.N / cfg.tile_n)
    tiles = gm * gn * max(1, cfg.split_k)
    w = max(1e-12, tiles / (hw.sm_count * max(1, cfg.max_blocks_per_sm)))
    full = math.ceil(w)
    return (full - w) / full


# (2) 절대량형 — 상한이 없다. 압축을 고려해야 할 수 있다
def edge_waste(p, hw, cfg) -> float:
    """타일이 형상 경계를 넘어 버려지는 일의 배수 - 1. 클수록 나쁘다.

    타일은 형상 밖으로 튀어나가도 그 부분을 전부 계산한다.
    M=1 에 128행 타일이면 일의 99.2% 가 버려진다 (값 127).
    출처: CUTLASS 문서, predication
    """
    gm = math.ceil(p.M / cfg.tile_m)
    gn = math.ceil(p.N / cfg.tile_n)
    return (gm * cfg.tile_m / p.M) * (gn * cfg.tile_n / p.N) - 1.0


# (3) 이진형 — 켜지면 자릿수가 달라지는 것
def has_spill(p, hw, cfg) -> float:
    """레지스터 스필이 있는가. 0 또는 1.

    레지스터가 SM 한계를 넘으면 로컬 메모리로 밀려난다. 로컬은
    물리적으로 DRAM 이고 mainloop 안에서 매 반복 접근한다.
    레지스터 접근이 1사이클이면 로컬은 수백 사이클이다.
    출처: CUDA C++ Best Practices Guide, "Register Pressure"
    """
    return 1.0 if cfg.spill_bytes > 0 else 0.0
```

셋의 차이가 요점입니다 — **비율 / 절대량 / 이진.** 재려는 물리가 어느
형태인지 먼저 정하고 식을 쓰세요.

**`rationale` 에 출처를 적을 수 있으면 적으세요.** 공개된 물리면 그럴 수
있고, 못 적겠으면 그 물리를 스스로 유도했다는 뜻이니 유도 과정을 쓰세요.
