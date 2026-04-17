import torch
import cv2
import numpy as np
import lpips
from pathlib import Path
import argparse
from tqdm import tqdm

from models.exposure_synthesizer import ExposureSynthesizer
from utils.dataset import HDREyeDataset
from utils.hdr import read_hdr, get_exif, read_exposure_time, measure_ev_range
from utils.metrics import compute_psnr, tensor_to_numpy, compute_lpips_tensor


def create_hdr_debevec(images, exposure_times):
    merge_debevec = cv2.createMergeDebevec()
    hdr = merge_debevec.process(images, times=exposure_times)
    return hdr


def load_exposure_times(ldr_dir, scene):
    exposures = []
    image_files = sorted(ldr_dir / scene.glob('*.JPG*'))
    for img_path in image_files:
        exif = get_exif(str(img_path))
        exposures.append(read_exposure_time(exif))
    return np.array(exposures, dtype=np.float32)


def load_ldr_images(ldr_dir, scene):
    images = []
    image_files = sorted(ldr_dir / scene.glob('*.JPG*'))
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        images.append(img)
    return images


def evaluate_exposure_synthesis(model, test_dataset, device, lpips_model, target_ev=2.7):
    model.eval()
    
    psnr_under = []
    psnr_over = []
    lpips_under = []
    lpips_over = []
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
            ldr_tensor, _, _ = test_dataset[i]
            ldr_tensor = ldr_tensor.unsqueeze(0).to(device)
            
            output = model(ldr_tensor)
            
            underexposed = output[0, :3]
            overexposed = output[0, 3:]
            
            from utils.hdr import apply_exposure_adjustment
            ldr_np = (ldr_tensor[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            target_under_np = apply_exposure_adjustment(ldr_np, -target_ev)
            target_over_np = apply_exposure_adjustment(ldr_np, target_ev)
            
            pred_under = (underexposed.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            pred_over = (overexposed.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            
            psnr_under.append(compute_psnr(target_under_np, pred_under))
            psnr_over.append(compute_psnr(target_over_np, pred_over))
            
            lpips_under.append(compute_lpips_tensor(
                underexposed.unsqueeze(0), 
                torch.from_numpy(target_under_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0,
                lpips_model, device
            ))
            lpips_over.append(compute_lpips_tensor(
                overexposed.unsqueeze(0),
                torch.from_numpy(target_over_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0,
                lpips_model, device
            ))
    
    results = {
        'underexposed': {'psnr': np.mean(psnr_under), 'lpips': np.mean(lpips_under)},
        'overexposed': {'psnr': np.mean(psnr_over), 'lpips': np.mean(lpips_over)}
    }
    return results


def evaluate_hdr_reconstruction(model, test_dataset, device, ldr_dir):
    model.eval()
    
    results = []
    
    test_scenes = ['C40', 'C41', 'C42', 'C43', 'C44', 'C45', 'C46']
    
    for scene in tqdm(test_scenes, desc="HDR Reconstruction"):
        scene_path = ldr_dir / scene
        
        ldr_files = sorted(scene_path.glob('*.JPG*'))
        
        original_hdr_path = test_dataset.hdr_dir / f"{scene}.hdr"
        if original_hdr_path.exists():
            original_hdr = read_hdr(str(original_hdr_path))
            original_dr = measure_ev_range(original_hdr)
        else:
            original_dr = 0
        
        exposure_times = []
        ldr_images = []
        for ldr_file in ldr_files:
            exif = get_exif(str(ldr_file))
            exposure_times.append(read_exposure_time(exif))
            img = cv2.imread(str(ldr_file))
            ldr_images.append(img)
        
        exposure_times = np.array(exposure_times, dtype=np.float32)
        
        merged_hdr = create_hdr_debevec(ldr_images, exposure_times)
        new_dr = measure_ev_range(merged_hdr)
        
        results.append({
            'scene': scene,
            'original_dr': original_dr,
            'new_dr': new_dr
        })
    
    return results


def print_results(synthesis_results, hdr_results):
    print("\n=== Exposure Synthesis Results ===")
    print(f"{'Method':<15} {'PSNR':<10} {'LPIPS':<10}")
    print(f"{'underexposed':<15} {synthesis_results['underexposed']['psnr']:<10.4f} {synthesis_results['underexposed']['lpips']:<10.4f}")
    print(f"{'overexposed':<15} {synthesis_results['overexposed']['psnr']:<10.4f} {synthesis_results['overexposed']['lpips']:<10.4f}")
    
    print("\n=== HDR Reconstruction Results ===")
    print(f"{'Obraz':<8} {'Dynamic Range Original':<25} {'Dynamic Range New':<20}")
    for r in hdr_results:
        print(f"{r['scene']:<8} {r['original_dr']:<25.4f} {r['new_dr']:<20.4f}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ExposureSynthesizer().to(device)
    model.load_state_dict(torch.load(args.model_path))
    
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    
    test_dataset = HDREyeDataset(args.data_dir, split='test')
    
    synthesis_results = evaluate_exposure_synthesis(
        model, test_dataset, device, lpips_model, args.ev
    )
    
    hdr_results = evaluate_hdr_reconstruction(
        model, test_dataset, device, test_dataset.bracketed_dir
    )
    
    print_results(synthesis_results, hdr_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to HDR-Eye dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--ev', type=float, default=2.7, help='Target EV for synthesis')
    args = parser.parse_args()
    
    main(args)
