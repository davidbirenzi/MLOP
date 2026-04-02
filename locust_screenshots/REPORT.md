# Locust load test — short report

**Setup:** Docker Compose + Nginx on `http://127.0.0.1:8000`, task `POST /predict` only.  
**Load:** 5 users, ramp 5 users/s (see `1.png`).

| API replicas | RPS | Median (ms) | 95%ile (ms) | 99%ile (ms) | Avg (ms) | Fails | Requests (snapshot) |
|-------------:|----:|------------:|------------:|------------:|---------:|------:|--------------------:|
| 1 | 2.4 | 170 | 290 | 1500 | 192 | 1 / 470 (~0.2%) | 470 |
| 2 | 2.2 | 170 | 270 | 1500 | 192 | 0 / 452 | 452 |
| 3 | 1.9 | 200 | 470 | 2100 | 242 | 0 / 468 | 468 |

**Takeaway:** Throughput stayed in the same band (~1.9–2.4 RPS) with 5 users—bottleneck is mostly model inference per request, not just raw container count. Two replicas matched one replica on median latency; three replicas showed higher tail latency (95th/99th) in this run, which often happens on one machine (CPU contention, Nginx + extra processes). Failures stayed at effectively zero except one request in the first run.

Screenshots: `1.png` (settings), `1.1.png` (1 replica), `2.2.png` (2 replicas), `3.3.png` (3 replicas).
