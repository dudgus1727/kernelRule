## 형태 예시 — ★ 다른 도메인의 것입니다

아래 셋은 **이 문제와 무관한 도메인**의 예시입니다.
물리를 어떻게 코드로 옮기는지, 설명을 어떻게 쓰는지, 크기를 어떻게
맞추는지만 보세요. **여기 나온 개념을 GEMM 에 옮기려 하지 마세요.**

```python
# (1) 비율형 — 0~1 로 정규화되고 물리적 상한이 있다
def branch_divergence_cost(warp_size, active_lanes) -> float:
    """워프 안에서 갈라지는 분기의 비용. 클수록 나쁘다.

    SIMT 는 워프 단위로 실행되므로 분기가 달라지면 양쪽을 순차 실행한다.
    활성 레인이 적을수록 나머지 레인이 논다.
    """
    return 1.0 - active_lanes / max(warp_size, 1)


# (2) 절대량형 — 상한이 없어 로그로 압축한다
def queue_backlog(arrival_rate, service_rate) -> float:
    """대기열에 쌓이는 정도. 클수록 나쁘다.

    도착이 처리보다 빠르면 무한히 쌓인다. 포화 근처에서 급격히
    나빠지므로 로그로 압축해 다른 항과 크기를 맞춘다.
    """
    rho = arrival_rate / max(service_rate, 1e-9)
    return math.log2(1.0 + rho / max(1.0 - min(rho, 0.999), 1e-9))


# (3) 이진형 — 켜지면 자릿수가 달라지는 것
def page_fault_present(working_set, ram_bytes) -> float:
    """워킹셋이 물리 메모리를 넘는가. 넘으면 자릿수가 달라진다.

    디스크 접근은 메모리보다 수만 배 느리므로 이진으로 충분하다.
    """
    return 1.0 if working_set > ram_bytes else 0.0
```

셋의 차이가 요점입니다 — **정규화 / 로그 압축 / 이진.** 재려는 물리가
어느 형태인지 먼저 정하고 식을 쓰세요.
