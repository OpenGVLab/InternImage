import torch
load_path = "/mnt/petrelfs/wangwenhai/workspace_swj/work_dir/sim/configs/pjlab/imagenet22k_30ep_init_from_pretrain_mm_1_bs304x120_resume384_lr3e_5_320gpus_lr1e_5_resume512_resume640.sh/checkpoint-latest.pth"
print(load_path)
model = torch.load(load_path, map_location=torch.device('cpu'))
model = model['model']

new_model = {}
for k, v in model.items():
    if k.startswith("dcnv3."):
        new_k = k.replace("dcnv3.", "")
        new_model[new_k] = v
        if "sampling_offsets" in new_k or "attention_weights" in new_k:
            n_levels = 1
            n_points = 8
            stage_id = int(new_k.split(".")[1])
            n_heads = [10, 20, 40, 80][stage_id]
            if "sampling_offsets" in new_k: # offset初始化为0
                if ".bias" in new_k:
                    v = v.reshape(n_heads, n_levels, n_points, 2)
                    part1 = v[:, :, :4, ...]
                    part2 = v[:, :, 4:, ...]
                    pad = torch.zeros(part2.shape)[:, :, :1, ...] * 5
                    v = torch.cat([part1, pad, part2], dim=2)
                    v = v.reshape(n_heads * n_levels * (n_points + 1) * 2)
                else:
                    in_dim, out_dim = v.shape
                    v = v.reshape(n_heads, n_levels, n_points, 2, out_dim)
                    part1 = v[:, :, :4, ...]
                    part2 = v[:, :, 4:, ...]
                    pad = torch.zeros(part2.shape)[:, :, :1, ...]
                    v = torch.cat([part1, pad, part2], dim=2)
                    v = v.reshape(n_heads * n_levels * (n_points + 1) * 2, out_dim)
            elif "attention_weights" in new_k: # attention weight初始化为-5
                if ".bias" in new_k:
                    v = v.reshape(n_heads, n_levels, n_points)
                    part1 = v[:, :, :4]
                    part2 = v[:, :, 4:]
                    pad = torch.ones(part2.shape)[:, :, :1] * -5
                    v = torch.cat([part1, pad, part2], dim=2)
                    v = v.reshape(n_heads * n_levels * (n_points + 1))
                else:
                    in_dim, out_dim = v.shape
                    v = v.reshape(n_heads, n_levels, n_points, out_dim)
                    part1 = v[:, :, :4, ...]
                    part2 = v[:, :, 4:, ...]
                    pad = torch.zeros(part2.shape)[:, :, :1, ...]
                    v = torch.cat([part1, pad, part2], dim=2)
                    v = v.reshape(n_heads * n_levels * (n_points + 1), out_dim)

            new_model[new_k] = v
            if "gamma" in new_k:
                print(new_k, v.shape)

save_path = "/mnt/petrelfs/share_data/chenzhe1/workspace/to_wwh/dcnv3_huge_pretrain/dcnv3_5_new2_huge_checkpoint-final-640.pth"
torch.save(new_model, save_path)