"""
Syntax and Import Verification Script

This script verifies that all GRPO-related Python files are syntactically correct
and that the import structure is valid (without actually importing heavy dependencies).
"""

import ast
import os
import sys
from pathlib import Path


def check_syntax(filepath):
    """Check if a Python file has valid syntax"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def check_file_structure():
    """Verify that all expected files exist"""
    expected_files = [
        'grpo_reward.py',
        'motion_constraint_executor.py',
        'spatiotemporal_reward.py',
        'train_grpo.py',
        'run_smoke_test.py',
        'GRPO_TECHNICAL_NOTES.md',
    ]

    missing = []
    for filename in expected_files:
        if not os.path.exists(filename):
            missing.append(filename)

    return missing


def analyze_grpo_implementation():
    """Analyze key aspects of the GRPO implementation"""
    print("\n" + "="*70)
    print("Analyzing GRPO Implementation Structure")
    print("="*70)

    checks = []

    # Check train_grpo.py
    print("\n[1] Analyzing train_grpo.py...")
    try:
        with open('train_grpo.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for key components
        key_components = {
            'GRPOTrainer class': 'class GRPOTrainer' in content,
            'group_sample method': 'def group_sample' in content,
            'compute_grpo_loss method': 'def compute_grpo_loss' in content,
            'log-prob computation method': (
                'def compute_log_probs' in content or
                'def compute_batch_log_probs' in content
            ),
            'disable_adapter usage': 'disable_adapter' in content,
            'set_adapter usage': 'set_adapter' in content,
            'Ratio clipping': 'torch.clamp' in content and 'ratio' in content.lower(),
            'KL divergence': 'kl_div' in content or 'kl_penalty' in content,
            'Gradient accumulation': 'backward()' in content,
            'Learning rate schedule': 'def get_lr' in content or 'warmup' in content.lower(),
        }

        for component, found in key_components.items():
            status = "[OK]  " if found else "[FAIL]"
            print(f"  {status} {component}")
            checks.append((f"train_grpo.py: {component}", found))

    except Exception as e:
        print(f"  [FAIL] Error analyzing train_grpo.py: {e}")

    # Check grpo_reward.py
    print("\n[2] Analyzing grpo_reward.py...")
    try:
        with open('grpo_reward.py', 'r', encoding='utf-8') as f:
            content = f.read()

        key_components = {
            'GRPORewardModel class': 'class GRPORewardModel' in content,
            'compute_reward method': 'def compute_reward' in content,
            'executor integration': 'MotionConstraintExecutor' in content,
            '_decode_motion_tokens': '_decode_motion_tokens' in content,
            '_encode_text': '_encode_text' in content,
            '_compute_matching_score': '_compute_matching_score' in content,
            'Cosine similarity': 'F.normalize' in content or 'cosine' in content.lower(),
            'Batch processing': 'batch_size' in content or 'List[str]' in content,
        }

        for component, found in key_components.items():
            status = "[OK]  " if found else "[FAIL]"
            print(f"  {status} {component}")
            checks.append((f"grpo_reward.py: {component}", found))

    except Exception as e:
        print(f"  [FAIL] Error analyzing grpo_reward.py: {e}")

    # Summary
    print("\n" + "="*70)
    total = len(checks)
    passed = sum(1 for _, found in checks if found)
    print(f"Implementation Completeness: {passed}/{total} ({100*passed/total:.1f}%)")
    print("="*70)

    return all(found for _, found in checks)


def main():
    print("\n")
    print("="*70)
    print(" "*20 + "GRPO Syntax Verification")
    print("")
    print("  Verifying code structure without running heavy imports")
    print("="*70)
    print("\n")

    # Step 1: Check file structure
    print("[Step 1/3] Checking file structure...")
    missing = check_file_structure()
    if missing:
        print(f"  [FAIL] Missing files: {', '.join(missing)}")
        return False
    else:
        print("  [OK] All expected files present")

    # Step 2: Check syntax
    print("\n[Step 2/3] Checking Python syntax...")
    python_files = [
        'grpo_reward.py',
        'motion_constraint_executor.py',
        'spatiotemporal_reward.py',
        'train_grpo.py',
        'run_smoke_test.py',
    ]

    all_valid = True
    for filepath in python_files:
        valid, error = check_syntax(filepath)
        if valid:
            print(f"  [OK] {filepath}")
        else:
            print(f"  [FAIL] {filepath}: {error}")
            all_valid = False

    if not all_valid:
        print("\n[FAIL] Syntax errors found. Please fix them before proceeding.")
        return False

    # Step 3: Analyze implementation
    print("\n[Step 3/3] Analyzing implementation structure...")
    implementation_complete = analyze_grpo_implementation()

    # Final summary
    print("\n" + "="*70)
    if all_valid and implementation_complete:
        print("[SUCCESS] VERIFICATION PASSED!")
        print("="*70)
        print("\nAll checks passed:")
        print("  [OK] All files present")
        print("  [OK] No syntax errors")
        print("  [OK] All key components implemented")
        print("\nNext steps:")
        print("  1. Set up Python environment and local model assets")
        print("  2. Run: python run_smoke_test.py")
        print("  3. Train: python train_grpo.py --sft-checkpoint <path>")
        return True
    else:
        print("[FAIL] VERIFICATION FAILED")
        print("="*70)
        print("\nPlease check the errors above and fix them.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
