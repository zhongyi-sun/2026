import os

full_path = "/home/sulci/JS/A_joyCODE/DEV_LIN/test_CST_MOTOHD/CST/ICP/transRotMat_tal_L_flip-R011160MHU_voirT_CST.tx"

def debug_path(p):
    parts = p.split('/')
    current = "/"
    for part in parts:
        if not part: continue
        current = os.path.join(current, part)
        exists = os.path.exists(current)
        print(f"{' [OK] ' if exists else '[MISS]'} {current}")
        if not exists:
            # If it's missing, let's see what IS there
            parent = os.path.dirname(current)
            if os.path.exists(parent):
                print(f"  --> Directory exists, but '{part}' not found.")
                print(f"  --> Contents of {parent}: {os.listdir(parent)[:5]}...")
            break

debug_path(full_path)
