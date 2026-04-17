import cv2
import numpy as np
import torch
from pathlib import Path
from utils.hdr import tone_map_reinhard, measure_ev_range, read_hdr, get_exif, read_exposure_time
from models.exposure_synthesizer import ExposureSynthesizer

FONT = cv2.FONT_HERSHEY_SIMPLEX
IMG_SIZE = (512, 512)

def tonemap_u8(hdr_rgb):
    ldr = tone_map_reinhard(hdr_rgb)
    ldr = np.nan_to_num(ldr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(ldr * 255.0, 0, 255).astype(np.uint8)

def add_side_label(img, text):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.putText(img, text, (5, 18), FONT, 0.45, (255, 255, 255), 1)
    return img

def load_ldr_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
    return img

def merge_debevec(rgb_images, times):
    bgr_list = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in rgb_images]
    merged_bgr = cv2.createMergeDebevec().process(bgr_list, times=times.astype(np.float32))
    merged = cv2.cvtColor(merged_bgr, cv2.COLOR_BGR2RGB)
    return merged.astype(np.float32)

def synth_under_over(model, base_rgb_u8, device):
    x = torch.from_numpy(base_rgb_u8).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x = x.to(device)
    with torch.no_grad():
        y = model(x)[0].cpu().numpy()
    y = np.clip(y, 0, 1).transpose(1, 2, 0)
    under = (y[:, :, :3] * 255).astype(np.uint8)
    over = (y[:, :, 3:] * 255).astype(np.uint8)
    return under, over

def load_original_bracketed(scene_path):
    ldr_files = sorted(scene_path.glob('*.JPG*'))
    images = []
    for f in ldr_files:
        try:
            exif = get_exif(str(f))
            exp_time = read_exposure_time(exif)
            img = load_ldr_rgb(f)
            images.append((img, exp_time))
        except:
            continue
    return images

def psnr_db(a, b, max_val):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse <= 0:
        return float('inf')
    return 20.0 * np.log10(max_val / np.sqrt(mse))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_baseline = ExposureSynthesizer().to(device)
state_baseline = torch.load('lab2/checkpoints_backup/model_epoch_50.pt', map_location=device)
model_baseline.load_state_dict(state_baseline)
model_baseline.eval()

model_synth = ExposureSynthesizer().to(device)
state_synth = torch.load('lab2/checkpoints/model_epoch_50.pt', map_location=device)
model_synth.load_state_dict(state_synth)
model_synth.eval()

compare_dir = Path('results/compare')
bracketed_dir = Path('lab2/data/HDREye/images/Bracketed_images')
hdr_dir = Path('lab2/data/HDREye/images/HDR')
scenes = ['C40', 'C41', 'C43']
EV = 2.7

row1_imgs = []
row2_imgs = []
row3_imgs = []
metrics_list = []

for scene in scenes:
    scene_path = bracketed_dir / scene
    all_images = load_original_bracketed(scene_path)
    
    sorted_imgs = sorted(all_images, key=lambda x: x[1])
    base_rgb, base_exp = sorted_imgs[len(sorted_imgs) // 2]
    
    hdr_path = hdr_dir / f'{scene}_HDR.hdr'
    gt_hdr = read_hdr(str(hdr_path))
    gt_hdr = cv2.resize(gt_hdr, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    
    synth_under_baseline, synth_over_baseline = synth_under_over(model_baseline, base_rgb, device)
    synth_factor = 2.0 ** EV
    synth_times = np.array([base_exp / synth_factor, base_exp, base_exp * synth_factor], dtype=np.float32)
    synth_hdr_baseline = merge_debevec([synth_under_baseline, base_rgb, synth_over_baseline], synth_times)
    
    synth_under_synth, synth_over_synth = synth_under_over(model_synth, base_rgb, device)
    synth_hdr_synth = merge_debevec([synth_under_synth, base_rgb, synth_over_synth], synth_times)
    
    gt_dr = measure_ev_range(gt_hdr)
    dr_baseline = measure_ev_range(synth_hdr_baseline)
    dr_synth = measure_ev_range(synth_hdr_synth)
    
    psnr_baseline = psnr_db(tonemap_u8(gt_hdr), tonemap_u8(synth_hdr_baseline), 255.0)
    psnr_synth = psnr_db(tonemap_u8(gt_hdr), tonemap_u8(synth_hdr_synth), 255.0)
    
    metrics_list.append({
        'scene': scene,
        'gt_dr': gt_dr,
        'psnr_baseline': psnr_baseline,
        'dr_baseline': dr_baseline,
        'psnr_synth': psnr_synth,
        'dr_synth': dr_synth
    })
    
    row1_imgs.append(add_side_label(tonemap_u8(gt_hdr), f"{scene} GT HDR (DR: {gt_dr:.1f} EV)"))
    row2_imgs.append(add_side_label(tonemap_u8(synth_hdr_baseline), f"{scene} Baseline (PSNR: {psnr_baseline:.1f} dB, DR: {dr_baseline:.1f} EV)"))
    row3_imgs.append(add_side_label(tonemap_u8(synth_hdr_synth), f"{scene} +Synth (PSNR: {psnr_synth:.1f} dB, DR: {dr_synth:.1f} EV)"))

r1 = np.concatenate(row1_imgs, axis=1)
r2 = np.concatenate(row2_imgs, axis=1)
r3 = np.concatenate(row3_imgs, axis=1)

label_h = 40
pad = np.full((20, r1.shape[1], 3), 255, dtype=np.uint8)
m1 = np.full((label_h, r1.shape[1], 3), (40, 40, 40), dtype=np.uint8)
m2 = np.full((label_h, r1.shape[1], 3), (40, 40, 40), dtype=np.uint8)
m3 = np.full((label_h, r1.shape[1], 3), (40, 40, 40), dtype=np.uint8)

cv2.putText(m1, "GT HDR (from HDR-Eye .hdr files)", (20, 27), FONT, 0.65, (255, 255, 255), 2)
cv2.putText(m2, "Baseline Model (no synthetic data)", (20, 27), FONT, 0.65, (255, 255, 255), 2)
cv2.putText(m3, "Trained Model (+ synthetic data)", (20, 27), FONT, 0.65, (255, 255, 255), 2)

montage = np.concatenate([m1, r1, pad, m2, r2, pad, m3, r3], axis=0)
montage_bgr = cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(compare_dir / 'final_comparison.png'), montage_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])

print(f"Zapisano: {compare_dir}/final_comparison.png ({montage.shape[1]}x{montage.shape[0]})")
print("\nMetryki (GT = plik .hdr z katalogu HDR-Eye):")
for m in metrics_list:
    print(f"  {m['scene']}: GT DR={m['gt_dr']:.2f}, Base PSNR={m['psnr_baseline']:.2f}, Synth PSNR={m['psnr_synth']:.2f}")
