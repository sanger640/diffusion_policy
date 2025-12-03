import zarr
import os

# Hardcode the path exactly as it appears in the error log
PATH = "/home/sanger/diffusion_policy/data/paper_implementation.zarr"

def check():
    print(f"Checking {PATH}...")
    
    # 1. Check if it exists
    if not os.path.exists(PATH):
        print("❌ Error: Path does not exist!")
        return
        
    if not os.path.isdir(PATH):
        print("❌ Error: Path is not a directory (Zarr must be a folder)!")
        return

    # 2. Try to open with Zarr
    try:
        root = zarr.open_group(PATH, mode='r')
        print("✅ Success! zarr.open_group works.")
        print(f"   Keys: {list(root.keys())}")
        
        # Check deep keys
        if 'data/obs/agent_view_image' in root:
            print("   ✅ Image data found.")
        else:
            print("   ❌ Image data MISSING in Zarr structure.")
            
    except Exception as e:
        print(f"❌ Failed to open with zarr: {e}")

    # 3. Check for trailing slashes or weirdness
    print(f"Ends with .zarr? {'Yes' if PATH.endswith('.zarr') else 'No'}")

if __name__ == "__main__":
    check()