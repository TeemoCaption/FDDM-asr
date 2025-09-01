# scripts/sanity_check_scheduler.py
import json, torch
from fddm.sched.diffusion_scheduler import DiscreteDiffusionScheduler

# 讀取配置檔案以取得 vocab_size，而非硬編碼路徑
import yaml
config_path = "configs/fddm_zhTW_base.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
K = int(config["data"]["vocab_size"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sched = DiscreteDiffusionScheduler(K=K, T=200, device=device)

B, L = 2, 6
x0 = torch.zeros(B, L, K, device=device); x0[..., 10] = 1.0  # 假的 one-hot
t  = torch.randint(low=1, high=200, size=(B,), device=device)

xt = sched.q_sample(x0, t)
assert torch.allclose(xt.sum(-1), torch.ones(B, L, device=device), atol=1e-5)

x0hat = (x0 + 0.05 * torch.rand_like(x0))
x0hat = x0hat / x0hat.sum(-1, keepdim=True)

post = sched.q_posterior(xt, x0hat, t)
assert torch.allclose(post.sum(-1), torch.ones(B, L, device=device), atol=1e-5)
print("Scheduler sanity check passed.")
