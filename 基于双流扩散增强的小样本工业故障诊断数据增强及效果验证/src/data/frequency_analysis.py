import argparse
import numpy as np


def calculate_fault_frequencies(fs, rpm, num_balls, diameter_ratio, contact_angle_deg, gear_teeth):
    fr = rpm / 60
    alpha = np.deg2rad(contact_angle_deg)
    
    BPFI = (num_balls / 2) * fr * (1 + diameter_ratio * np.cos(alpha))
    BPFO = (num_balls / 2) * fr * (1 - diameter_ratio * np.cos(alpha))
    GMF = gear_teeth * fr
    
    return fr, BPFI, BPFO, GMF


def main():
    parser = argparse.ArgumentParser(description='Calculate fault characteristic frequencies')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--fs', type=int, default=12000, help='Sampling frequency')
    parser.add_argument('--rpm', type=int, default=1500, help='Rotational speed')
    parser.add_argument('--num_balls', type=int, default=8, help='Number of balls')
    parser.add_argument('--diameter_ratio', type=float, default=0.206, help='d/D ratio')
    parser.add_argument('--contact_angle', type=float, default=0, help='Contact angle in degrees')
    parser.add_argument('--gear_teeth', type=int, default=32, help='Number of gear teeth')
    args = parser.parse_args()
    
    fr, BPFI, BPFO, GMF = calculate_fault_frequencies(
        args.fs, args.rpm, args.num_balls, args.diameter_ratio, args.contact_angle, args.gear_teeth
    )
    
    print(f"Rotating frequency fr: {fr:.2f} Hz")
    print(f"BPFI: {BPFI:.2f} Hz")
    print(f"BPFO: {BPFO:.2f} Hz")
    print(f"GMF : {GMF:.2f} Hz")
    print(f"Rotor imbalance frequency: {fr:.2f} Hz")
    
    print("\n对应论文中的理论值：")
    print("| 故障类型 | 理论频率 |")
    print("|---|---:|")
    print(f"| 轴承内圈故障 | {BPFI:.1f} Hz |")
    print(f"| 轴承外圈故障 | {BPFO:.1f} Hz |")
    print(f"| 齿轮磨损 | {GMF:.1f} Hz |")
    print(f"| 转子不平衡 | {fr:.1f} Hz |")


if __name__ == '__main__':
    main()