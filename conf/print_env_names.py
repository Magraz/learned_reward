from pathlib import Path

if __name__ == "__main__":
    env_dir = Path(__file__).resolve().parent / "env"
    print(" ".join(f'"{path.stem}"' for path in sorted(env_dir.glob("*.yaml"))))
    print("\n")
    print(" ".join(f'"{path.stem}",' for path in sorted(env_dir.glob("*.yaml"))))
