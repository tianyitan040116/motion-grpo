"""
可视化生成的动作
用法: python visualize_motion.py --motion test_walk.npy --output test_walk.gif --text "a person walks forward"
"""
import numpy as np
import torch
import argparse
from utils.motion_utils import recover_from_ric, plot_3d_motion

# HumanML3D kinematic tree
t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion', type=str, required=True, help='Motion .npy file')
    parser.add_argument('--output', type=str, default='motion.gif', help='Output video file')
    parser.add_argument('--text', type=str, default='Generated Motion', help='Title text')
    parser.add_argument('--fps', type=int, default=20, help='FPS for video')
    args = parser.parse_args()

    print(f"Loading motion from: {args.motion}")
    motion_data = np.load(args.motion)
    print(f"Motion shape: {motion_data.shape}")

    motion_tensor = torch.from_numpy(motion_data).float()

    joints_3d = recover_from_ric(motion_tensor, joints_num=22)
    joints_3d_np = joints_3d.numpy()

    print(f"3D joints shape: {joints_3d_np.shape}")

    print(f"\nGenerating animation: {args.output}")
    print(f"Title: {args.text}")
    print("This may take a few minutes...")

    plot_3d_motion(
        args.output,
        t2m_kinematic_chain,
        joints_3d_np,
        args.text,
        fps=args.fps
    )

    print(f"\nAnimation saved to: {args.output}")

if __name__ == '__main__':
    main()
