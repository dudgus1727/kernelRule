# 진단 리포트 — dev-cu124-r000-handwritten

**재현** `python3 -m kernelrule.report.diagnostic` 경로가 없다 — `report/diagnostic.py::build_report` 를 손으로 불러 만든 것이다. ⚠️ 이 리포트의 블록 3.5 는 **오염된 상태**였다 (D-28).

> 개발용 표(dev-cu124, CUDA 12.4/호스트)다. 성능 수치를 인용하지 마라.
> 학습 분할(M<=2048, 50형상)만으로 만들어졌다. 홀드아웃 점수는 들어 있지 않다.

## 블록 1 — 하드웨어 사실
```
GPU: NVIDIA RTX A6000 (sm_86)
  SM 84개 / 블록당 smem 101376B
  SM당 최대 1536 스레드 / 레지스터 65536
  L2 6MB
  실효 116.1 TFLOP/s / 729.7 GB/s
  ridge point 159.1 FLOP/byte

실행 모델:
  CTA가 SM에 배분되며 마지막 wave에서 SM 일부가 유휴.
  타일은 형상 경계를 넘어도 그 부분을 전부 계산한다 — M=1에 128행
    타일이면 일의 99.2%가 버려진다.
  split-K는 K를 나눠 타일 수를 늘리되 리덕션 비용이 추가된다.
  serial split-K는 파티션마다 fp16으로 D를 왕복한다 (정밀도 손실).
  parallel split-K는 부분합 M*N*sk개를 DRAM에 쓰고 다시 읽는다.
  stages=2(MmaPipelined)와 stages>=3(multistage)은 다른 커널 계열이다.
  alignment가 16바이트를 못 맞추면 cp.async를 못 써서 2단만 가능하다.

측정의 한계:
  시간은 CUDA 이벤트 타이머의 눈금(1.024us) 단위로만
    기록된다. 그보다 작은 차이는 **측정으로 구분할 수 없다.**
  짧은 커널일수록 그 눈금이 상대적으로 크다 —
    14us에서 한 눈금이 7.3%, 1.3ms에서 0.08%.
```

## 블록 2 — 현재 규칙
가중치는 수치 최적화기가 맞춘 값이다: [1.013, 0.565, 0.132, 3.201, 0.747, 0.583, 0.512]
```python
def score(f, p, hw, w):
    s  = np.log2(f.traffic_amplification) * w[0]
    s = s + f.sm_idle_cost * w[1]
    s = s + f.smem_pressure * w[2]
    s = s + f.has_spill * w[3]
    s = s + f.split_k_cost * w[4]
    s = s + f.pipeline_warmup_frac * w[5]
    if p.is_memory_bound:
        s = s + np.log2(f.traffic_amplification) * w[6]
    return s
```

현재 규칙에는 다음 가설들이 반영되어 있다:
  H0 (초기): 트래픽 최소화가 주항. 로그로 포화시킨다
  H1 (초기): wave 양자화는 비선형(1/(1-tail))이어야 크기가 나온다
  H2 (초기): 스필은 이진 벌점으로 충분하다

## 블록 3 — 체제별 regret 분해
```
전체 regret@1 1.1947  (@3 1.1690  @5 1.1674  @10 1.1658)
정답 적중 hit@1 0.000  hit@3 0.180  (정답 = 최적 대비 노이즈 바닥 2시그마 이내)
  regret 은 낮은데 hit 이 0 이면 **아깝게 빗나가는 것이 아니라
  구조적으로 다른 곳을 짚는 것**이다 — 가중치 조정이 아니라 항이 필요하다.

층화 — 이 분할에서는 **난이도 층화가 더 크게 달라진다** (0.1651 vs 0.0007)
  t_sol >= 0.5ms   1.1953   (9형상)
  t_sol <  0.5ms   1.1946   (41형상)   격차 -0.0007
  난이도 상 / 하    1.2802 / 1.1150   격차 +0.1651

체제                           형상   regret   최악 형상
K <= 1024 (짧은 mainloop)       8   1.3812   128x4096x256 (2.091)
compute-bound                30   1.2385   512x512x512 (1.545)
waves < 1                    15   1.2154   128x4096x256 (2.091)
t_sol >= 0.5ms (김)            9   1.1953   2048x4096x11008 (1.275)
t_sol < 0.5ms (짧음)           41   1.1946   128x4096x256 (2.091)
waves 1~4                    25   1.1739   768x4096x4096 (1.420)
memory-bound                 20   1.1319   128x4096x256 (2.091)
waves > 8                     4   1.1248   1024x11008x4096 (1.200)
```

## 블록 3.5 — 표 구조 관찰
개별 사례로는 보이지 않는 패턴이다. **학습 분할에서만 계산했다** (§12.3).

> ★ **이 블록은 두 번 깎였다.** (1) 2026-08-22: 전수 66형상에서 계산되던 것을
> 학습 분할로 (D-28). (2) 2026-08-26: **축을 지목하는 줄을 전부 삭제**
> — "스필 커널이 최적으로 뽑힌 형상 0개", "stages=2 가 최적인 형상 2개",
> "GBDT 가 크게 의존한 축: ..." 은 전부 **정답 요약**이었다 (§12.3b).
> 남은 것은 "여지가 얼마나 되는가" 뿐이고, 어느 축을 봐야 하는지는
> 하나도 알려주지 않는다.

```
학습 분할 46형상에서만 계산했다 (§12.3).
고정 config 하나로 얼마나 가는가 (형상 무관):  top-1 1.122   top-3 1.116   top-8 1.030
  빠른 체제(SOL<0.5ms) 34형상만: top-1 1.158
정답이 하나로 정해지지 않는 형상(노이즈 안 동률): 44/46개
  동률 폭 중앙값 4개, 최대 522개
★ 제시하지 못한 관찰 1건 (지지 형상 15개 미만이거나 컬럼 부재). 조용히 빠지지 않는다 (§26.4).
```

## 블록 4 — 사례
**선택 vs 최적을 나란히 본다.** 주변 config 는 최적이 뾰족한지
넓은지 알려준다 — 넓으면 정확히 맞추라는 뜻이 아니다.

### 사례 #1  128x4096x256  [t_sol < 0.5ms (짧음)] 
```
규칙 선택: tb128x256x32 w64x64 st2 swhz1 sk4par           ->     23.55us  (regret 2.091)
실제 최적: tb64x128x32 w32x64 st5 swhz1 sk1ser            ->     11.26us

격차 = 노이즈 바닥의 **12.0배**

피처(차이 큰 순)                             선택           최적  규칙에서
log_workspace_bytes               22.0000       0.0000  ★ 미사용
is_two_stage                       1.0000       0.0000  ★ 미사용
split_k_cost                       0.5000       0.0000  사용 중
reg_pressure                       0.9453       0.2812  ★ 미사용
log_mainloop_iters                 1.0000       3.0000  ★ 미사용
pipeline_warmup_frac               1.0000       0.6250  사용 중
traffic_amplification              1.4545       2.9091  사용 중
smem_pressure                      0.4848       0.6061  사용 중

같은 형상 상위 5개 실측 (최적 대비 노이즈 바닥 배수):
      11.26us          (최적)  tb64x64x32 w16x64 st3 swid1 sk1ser
      11.26us          (최적)  tb64x128x32 w64x32 st5 swhz1 sk1ser
      11.26us          (최적)  tb64x128x32 w64x32 st5 swid8 sk1ser
      11.26us          (최적)  tb64x128x32 w32x64 st5 swid8 sk1ser
      11.26us          (최적)  tb64x128x32 w64x32 st5 swid1 sk1ser

난이도 1.99   노이즈 바닥 9.091%   구분 불가능한 정답 522/5775개
```

### 사례 #2  128x4096x1024  [t_sol < 0.5ms (짧음)] 
```
규칙 선택: tb128x256x32 w64x64 st2 swhz1 sk4par           ->     36.86us  (regret 1.565)
실제 최적: tb128x64x32 w16x64 st5 swhz1 sk1ser            ->     23.55us

격차 = 노이즈 바닥의 **13.0배**

피처(차이 큰 순)                             선택           최적  규칙에서
log_workspace_bytes               22.0000       0.0000  ★ 미사용
is_two_stage                       1.0000       0.0000  ★ 미사용
split_k_cost                       0.5000       0.0000  사용 중
reg_pressure                       0.9453       0.3750  ★ 미사용
pipeline_warmup_frac               0.2500       0.1562  사용 중
traffic_amplification              1.4545       2.9091  사용 중
log_mainloop_iters                 3.0000       5.0000  ★ 미사용
smem_pressure                      0.4848       0.6061  사용 중

같은 형상 상위 5개 실측 (최적 대비 노이즈 바닥 배수):
      23.55us          (최적)  tb128x64x32 w64x32 st6 swhz1 sk1ser
      23.55us          (최적)  tb128x64x32 w32x32 st4 swid2 sk1ser
      23.55us          (최적)  tb128x64x64 w16x64 st3 swid8 sk1ser
      23.55us          (최적)  tb128x64x32 w32x32 st6 swid2 sk1ser
      23.55us          (최적)  tb64x128x32 w16x64 st5 swid2 sk1ser

난이도 1.83   노이즈 바닥 4.348%   구분 불가능한 정답 364/15015개
```

### 사례 #3  2048x4096x11008  [t_sol >= 0.5ms (김)] 
```
규칙 선택: tb128x256x32 w64x64 st2 swhz1 sk1ser           ->   2177.09us  (regret 1.275)
실제 최적: tb128x128x32 w64x64 st3 swid8 sk6ser           ->   1708.03us

격차 = 노이즈 바닥의 **416.7배**

피처(차이 큰 순)                             선택           최적  규칙에서
is_two_stage                       1.0000       0.0000  ★ 미사용
tile_aspect_imbalance              1.0000       0.0000  ★ 미사용
sm_idle_cost                       0.3125       0.0391  사용 중
tail_waste                         0.2381       0.0376  ★ 미사용
reg_pressure                       0.9453       0.4453  ★ 미사용
split_k_cost                       0.0000       0.6462  사용 중
pipeline_warmup_frac               0.0058       0.0523  사용 중
waves                              3.0476      18.2857  ★ 미사용

같은 형상 상위 5개 실측 (최적 대비 노이즈 바닥 배수):
    1708.03us          (최적)  tb128x128x32 w64x64 st3 swid8 sk6ser
    1709.06us       +0.9시그마  tb128x128x32 w64x64 st3 swid4 sk6ser
    1711.10us       +2.7시그마  tb128x128x32 w64x64 st3 swid2 sk6ser
    1713.15us       +4.5시그마  tb128x128x32 w64x64 st3 swid8 sk8ser
    1714.18us       +5.5시그마  tb128x128x32 w64x64 st3 swid4 sk8ser

난이도 1.53   노이즈 바닥 0.066%   구분 불가능한 정답 2/17325개
```

### 사례 #4  1024x4096x16384  [t_sol >= 0.5ms (김)] 
```
규칙 선택: tb128x256x32 w64x64 st2 swhz1 sk1ser           ->   1616.35us  (regret 1.264)
실제 최적: tb128x128x32 w64x64 st3 swid2 sk12ser          ->   1278.98us

격차 = 노이즈 바닥의 **329.5배**

피처(차이 큰 순)                             선택           최적  규칙에서
is_two_stage                       1.0000       0.0000  ★ 미사용
tile_aspect_imbalance              1.0000       0.0000  ★ 미사용
sm_idle_cost                       0.3125       0.0391  사용 중
tail_waste                         0.2381       0.0376  ★ 미사용
reg_pressure                       0.9453       0.4453  ★ 미사용
split_k_cost                       0.0000       0.8962  사용 중
pipeline_warmup_frac               0.0039       0.0703  사용 중
waves                              1.5238      18.2857  ★ 미사용

같은 형상 상위 5개 실측 (최적 대비 노이즈 바닥 배수):
    1278.98us          (최적)  tb128x128x32 w64x64 st3 swid2 sk12ser
    1280.00us       +1.0시그마  tb128x128x32 w64x64 st3 swid8 sk12ser
    1281.02us       +2.0시그마  tb128x128x32 w64x64 st3 swid4 sk12ser
    1281.02us       +2.0시그마  tb128x128x32 w64x64 st3 swid1 sk12ser
    1285.12us       +6.0시그마  tb128x128x32 w64x64 st3 swid4 sk8ser

난이도 1.44   노이즈 바닥 0.080%   구분 불가능한 정답 2/17325개
```

### 사례 #5  512x512x512  [compute-bound] 
```
규칙 선택: tb128x128x32 w64x32 st2 swhz1 sk4par           ->     17.41us  (regret 1.545)
실제 최적: tb64x64x32 w16x64 st4 swid1 sk1ser             ->     11.26us

격차 = 노이즈 바닥의 **6.0배**

피처(차이 큰 순)                             선택           최적  규칙에서
log_workspace_bytes               21.0000       0.0000  ★ 미사용
is_two_stage                       1.0000       0.0000  ★ 미사용
split_k_cost                       0.5000       0.0000  사용 중
waves                              0.7619       0.2540  ★ 미사용
reg_pressure                       0.5781       0.2148  ★ 미사용
pipeline_warmup_frac               0.5000       0.2500  사용 중
sm_idle_cost                       0.3125       2.9375  사용 중
tail_waste                         0.2381       0.7460  ★ 미사용

같은 형상 상위 5개 실측 (최적 대비 노이즈 바닥 배수):
      11.26us          (최적)  tb64x64x32 w32x32 st4 swid2 sk1ser
      11.26us          (최적)  tb64x64x32 w32x32 st6 swid8 sk1ser
      11.26us          (최적)  tb64x64x32 w32x32 st7 swid8 sk1ser
      11.26us          (최적)  tb64x64x64 w32x32 st3 swid2 sk1ser
      11.26us          (최적)  tb64x64x64 w16x64 st3 swid1 sk1ser

난이도 1.82   노이즈 바닥 9.091%   구분 불가능한 정답 270/10395개
```

### 사례 #6  768x4096x4096  [compute-bound] 
```
규칙 선택: tb128x256x32 w64x64 st2 swhz1 sk2par           ->    363.52us  (regret 1.420)
실제 최적: tb128x128x32 w64x64 st3 swid1 sk3ser           ->    256.00us

격차 = 노이즈 바닥의 **105.0배**

피처(차이 큰 순)                             선택           최적  규칙에서
log_workspace_bytes               23.5850       0.0000  ★ 미사용
is_two_stage                       1.0000       0.0000  ★ 미사용
tile_aspect_imbalance              1.0000       0.0000  ★ 미사용
reg_pressure                       0.9453       0.4453  ★ 미사용
sm_idle_cost                       0.3125       0.1667  사용 중
tail_waste                         0.2381       0.1429  ★ 미사용
pipeline_warmup_frac               0.0312       0.0703  사용 중
split_k_cost                       0.2500       0.3962  사용 중

같은 형상 상위 5개 실측 (최적 대비 노이즈 바닥 배수):
     256.00us          (최적)  tb128x128x32 w64x64 st3 swid1 sk3ser
     256.00us          (최적)  tb128x128x32 w64x64 st3 swid2 sk3ser
     257.02us       +1.0시그마  tb128x128x32 w64x64 st3 swid8 sk3ser
     257.02us       +1.0시그마  tb128x128x32 w64x64 st3 swid4 sk3ser
     261.12us       +5.0시그마  tb128x128x32 w64x64 st3 swhz1 sk3ser

난이도 1.69   노이즈 바닥 0.400%   구분 불가능한 정답 4/15015개
```

### 사례 #7  256x11008x4096  [waves 1~4] 
```
규칙 선택: tb128x256x32 w64x64 st2 swhz1 sk2par           ->    360.45us  (regret 1.419)
실제 최적: tb128x128x32 w64x64 st3 swid2 sk4ser           ->    253.95us

격차 = 노이즈 바닥의 **104.0배**

피처(차이 큰 순)                             선택           최적  규칙에서
log_workspace_bytes               23.4263       0.0000  ★ 미사용
is_two_stage                       1.0000       0.0000  ★ 미사용
tile_aspect_imbalance              1.0000       0.0000  ★ 미사용
reg_pressure                       0.9453       0.4453  ★ 미사용
sm_idle_cost                       0.4651       0.2209  사용 중
tail_waste                         0.3175       0.1810  ★ 미사용
pipeline_warmup_frac               0.0312       0.0938  사용 중
waves                              2.0476       4.0952  ★ 미사용

같은 형상 상위 5개 실측 (최적 대비 노이즈 바닥 배수):
     253.95us          (최적)  tb128x128x32 w64x64 st3 swid2 sk4ser
     254.98us       +1.0시그마  tb128x128x32 w64x64 st3 swid4 sk4ser
     254.98us       +1.0시그마  tb128x128x32 w64x64 st3 swid2 sk3ser
     256.00us       +2.0시그마  tb128x128x32 w64x64 st3 swid1 sk4ser
     257.02us       +3.0시그마  tb128x128x32 w64x64 st3 swid8 sk4ser

난이도 1.61   노이즈 바닥 0.403%   구분 불가능한 정답 3/15015개
```

### 사례 #8  1024x11008x4096  [waves > 8] 
```
규칙 선택: tb128x256x32 w64x64 st2 swhz1 sk1ser           ->   1046.53us  (regret 1.200)
실제 최적: tb128x128x32 w64x64 st3 swid1 sk3ser           ->    872.45us

격차 = 노이즈 바닥의 **170.0배**

피처(차이 큰 순)                             선택           최적  규칙에서
is_two_stage                       1.0000       0.0000  ★ 미사용
tile_aspect_imbalance              1.0000       0.0000  ★ 미사용
sm_idle_cost                       0.2209       0.0581  사용 중
tail_waste                         0.1810       0.0549  ★ 미사용
reg_pressure                       0.9453       0.4453  ★ 미사용
split_k_cost                       0.0000       0.3962  사용 중
pipeline_warmup_frac               0.0156       0.0703  사용 중
waves                              4.0952      12.2857  ★ 미사용

같은 형상 상위 5개 실측 (최적 대비 노이즈 바닥 배수):
     872.45us          (최적)  tb128x128x32 w64x64 st3 swid4 sk3ser
     872.45us          (최적)  tb128x128x32 w64x64 st3 swid1 sk3ser
     873.47us       +1.0시그마  tb128x128x32 w64x64 st3 swid2 sk3ser
     873.47us       +1.0시그마  tb128x128x32 w64x64 st3 swid8 sk3ser
     875.52us       +3.0시그마  tb128x128x32 w64x64 st3 swid2 sk4ser

난이도 1.75   노이즈 바닥 0.117%   구분 불가능한 정답 4/15015개
```

### 사례 #9  2048x11008x4096  [waves > 8] 
```
규칙 선택: tb128x256x32 w64x64 st2 swhz1 sk1ser           ->   1883.14us  (regret 1.111)
실제 최적: tb128x128x32 w64x64 st3 swid8 sk2ser           ->   1695.74us

격차 = 노이즈 바닥의 **167.3배**

피처(차이 큰 순)                             선택           최적  규칙에서
is_two_stage                       1.0000       0.0000  ★ 미사용
tile_aspect_imbalance              1.0000       0.0000  ★ 미사용
sm_idle_cost                       0.0988       0.0378  사용 중
tail_waste                         0.0899       0.0364  ★ 미사용
reg_pressure                       0.9453       0.4453  ★ 미사용
split_k_cost                       0.0000       0.2500  사용 중
pipeline_warmup_frac               0.0156       0.0469  사용 중
waves                              8.1905      16.3810  ★ 미사용

같은 형상 상위 5개 실측 (최적 대비 노이즈 바닥 배수):
    1695.74us          (최적)  tb128x128x32 w64x64 st3 swid8 sk2ser
    1696.77us       +0.9시그마  tb128x128x32 w64x64 st3 swid4 sk2ser
    1697.79us       +1.8시그마  tb128x128x32 w64x64 st3 swid2 sk2ser
    1698.82us       +2.7시그마  tb128x128x32 w64x64 st3 swid1 sk2ser
    1715.20us      +17.4시그마  tb128x128x32 w64x64 st3 swid8 sk3ser

난이도 1.80   노이즈 바닥 0.066%   구분 불가능한 정답 3/15015개
```

### 사례 #10  1x12288x4096  [잘 맞춘 사례] ★ 잘 맞춤
```
규칙 선택: tb32x128x32 w16x64 st3 swhz1 sk2par            ->    158.72us  (regret 1.013)
실제 최적: tb32x128x32 w16x64 st3 swid1 sk2ser            ->    156.67us

격차 = 노이즈 바닥의 **2.0배**

피처(차이 큰 순)                             선택           최적  규칙에서
log_workspace_bytes               15.5850       0.0000  ★ 미사용
reg_pressure                       0.2070       0.2109  ★ 미사용
log_inst_total                     9.8734       9.8857  ★ 미사용

같은 형상 상위 5개 실측 (최적 대비 노이즈 바닥 배수):
     156.67us          (최적)  tb32x128x32 w32x32 st4 swid4 sk1ser
     156.67us          (최적)  tb32x128x64 w32x32 st3 swid1 sk3ser
     156.67us          (최적)  tb32x128x32 w32x32 st4 swid2 sk1ser
     156.67us          (최적)  tb32x128x32 w16x64 st3 swid8 sk4ser
     156.67us          (최적)  tb32x128x32 w32x32 st3 swid2 sk6ser

난이도 1.12   노이즈 바닥 0.654%   구분 불가능한 정답 132/15015개
```

### 사례 #11  8x12288x4096  [잘 맞춘 사례] ★ 잘 맞춤
```
규칙 선택: tb32x128x32 w16x64 st3 swhz1 sk2par            ->    158.72us  (regret 1.013)
실제 최적: tb32x128x32 w16x64 st3 swhz1 sk2ser            ->    156.67us

격차 = 노이즈 바닥의 **2.0배**

피처(차이 큰 순)                             선택           최적  규칙에서
log_workspace_bytes               18.5850       0.0000  ★ 미사용

같은 형상 상위 5개 실측 (최적 대비 노이즈 바닥 배수):
     156.67us          (최적)  tb32x128x32 w16x64 st3 swid1 sk2ser
     156.67us          (최적)  tb32x128x32 w32x32 st4 swid1 sk1ser
     156.67us          (최적)  tb32x128x32 w32x32 st3 swid8 sk2ser
     156.67us          (최적)  tb32x128x32 w16x64 st3 swid8 sk2ser
     156.67us          (최적)  tb32x128x32 w32x32 st3 swid2 sk2ser

난이도 1.14   노이즈 바닥 0.654%   구분 불가능한 정답 19/15015개
```

## 블록 5 — 실패 이력
**같은 아이디어를 반복하지 마라.**
```
r0    made_worse   - -> 1.758   traffic_amplification 을 선형으로 사용
r0    no_effect    1.172 -> 1.172   if p.is_memory_bound 에서 점수 전체에 스칼라 곱
```