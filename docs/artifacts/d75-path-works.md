# 경로가 신호를 옮긴다 — 요구 빈도와 별개로 성립한다 (2026-08-28)

> **상태**: 확정. **3차 실행 결과와 무관하다**
> **출처** [d75-run2.md](d75-run2.md) 의 2차 실행 (3시드 x 4라운드)
> **재현** `python3 experiments/d75_observe.py runs/f1pipe-F3-d75run2-s{0,1,2}`
> **표** dev-cu124. 성능 수치는 인용하지 않는다

2차 실행에서 **요구 빈도는 눌려 있었다**(D-80). 그러나 실제로 나온
요구 두 건에 대해서는 경로가 **끝까지** 돌았고, 그 결과는 빈도와
독립적으로 읽을 수 있다.

## 끝까지 돌았다

```
요구 2건 -> FeatureWriter 2회 -> §8.3 검증 통과 2/2 -> 등록 2/2
★ 되돌아간 Analyst 가 그 축을 언급   2/2
아카이브 사용   s0 2/7 규칙,  s1 3/7 규칙 (s1 은 **최선 규칙**에도)
요구 문장 누출  0/2
상한 초과로 버린 요구  0건
```

## ★ 만든 축이 옛 303건의 상위 주제와 겹친다

```python
# s0 r0 — 요구: "split-K가 추가하는 CTA 병렬성으로 인해 SM 유휴와
#              마지막 wave 손실이 얼마나 줄어드는지"
def split_k_wave_idle_delta(p, hw, cfg) -> float:
    """Change in final-wave idle capacity caused by split-K; larger is worse."""
    gm = math.ceil(p.M / max(cfg.tile_m, 1e-9))
    gn = math.ceil(p.N / max(cfg.tile_n, 1e-9))
    base_ctas = gm * gn
    split_ctas = base_ctas * max(cfg.split_k, 1)
    wave_cap = max(hw.sm_count * max(cfg.max_blocks_per_sm, 1), 1)
    ...

# s1 r2 — 요구: "출력 형상 M/N 종횡비와 타일 종횡비가 얼마나 정렬되는지"
def output_tile_aspect_misalignment(p, hw, cfg) -> float:
    out_aspect = max(p.M, 1e-9) / max(p.N, 1e-9)
    tile_aspect = max(cfg.tile_m, 1e-9) / max(cfg.tile_n, 1e-9)
    return abs(math.log2(max(out_aspect, 1e-9) / max(tile_aspect, 1e-9)))
```

옛 303건의 주제 순위와 대조하면:

| 옛 요구 주제 | 건수 | 이번에 만들어졌나 |
|---|---:|---|
| wave/CTA 절대량 | 82 | (부분) `split_k_wave_idle_delta` 가 wave 용량을 절대량으로 쓴다 |
| L2 재사용 이득 | 64 | 아직 |
| 파이프라인 계열 | 51 | 아직 |
| **split-K 의 이득** | **48** | ✅ `split_k_wave_idle_delta` |
| launch 고정 비용 | 21 | 아직 |
| warp 수준 | 15 | ★ `cfg.ext` — 의도적 금지 |
| **M/N 비대칭** | **13** | ✅ `output_tile_aspect_misalignment` |

**`split_k_wave_idle_delta` 는 "우리 피처는 비용만 재고 이득을 안
잰다" 를 그대로 메운다.** 옛 `split_k_cost` 는 리덕션 **비용**만 재고
CTA 병렬성 **이득**이 없었다. 48건이 그것을 요구했고, 경로가 생기자
그것이 만들어졌다.

## 왜 이것이 빈도와 별개인가

```
빈도가 낮다   = Analyst 가 "새 축이 필요하다" 고 **덜 말한다**
경로가 돈다   = 말했을 때 그것이 **실제 피처가 된다**
```

빈도는 프롬프트 문구에 달렸고(D-80/D-81), **말했을 때 무슨 일이
일어나는가**는 그것과 무관하다. 2건이 작은 표본인 것은 맞지만,
**2/2 가 검증을 통과하고 2/2 가 되돌아간 Analyst 에게 읽혔고 1/2 가
최선 규칙에 들어갔다** — 어느 단계에서도 끊기지 않았다.

⚠️ **"경로가 성능을 올린다" 는 주장이 아니다.** 4라운드 3시드로는 잴
수 없고, 재려면 시드 폭(σ=0.0124)을 넘는 차이가 필요하다.

## 조건 1(리포트 차단)이 모델의 습관과 어긋나지 않는다

```
옛 303건의 표 정보 누출   0/303   (안내 없이도 물리 언어로 썼다)
2차 2건                  0/2
```

FeatureWriter 에게 리포트를 안 주는 설계가 **모델에게 부자연스러운
요구가 아니다.** 그래서 3차에서 그 안내 문구를 빼도 (D-81 복원) 누출
위험이 커진다고 볼 근거가 없다 — 다만 **3차에서 다시 센다.**
