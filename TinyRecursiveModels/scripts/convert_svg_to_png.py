import os
import re
import glob
import subprocess
import xml.etree.ElementTree as ET
import tempfile
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Register namespace to output clean tags without ns0 prefix
ET.register_namespace('', "http://www.w3.org/2000/svg")

TASKS_DIR = r"d:\trm_c2\TinyRecursiveModels\reports\arc_task_atlas\tasks"

def get_browser_path():
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if os.path.exists(chrome_path):
        return chrome_path
    elif os.path.exists(edge_path):
        return edge_path
    else:
        raise RuntimeError("Neither Chrome nor Edge was found at the expected paths.")

def process_file(svg_path, browser_path):
    try:
        basename = os.path.basename(svg_path)
        png_name = basename.replace(".svg", ".png")
        png_path = os.path.join(os.path.dirname(svg_path), png_name)
        
        # Load the SVG XML
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Extract height
        height_str = root.attrib.get('height')
        height = int(height_str) if height_str else 940
        
        # Set root width to "960"
        root.attrib['width'] = "960"
        
        # Change viewBox width value to 960 (e.g. from "0 0 1440 940" to "0 0 960 940")
        viewbox = root.attrib.get('viewBox')
        if viewbox:
            parts = re.split(r'[\s,]+', viewbox.strip())
            if len(parts) >= 4:
                parts[2] = "960"
                root.attrib['viewBox'] = " ".join(parts)
                
        # Find all <rect> elements and if their width is "1440", set it to "960"
        for elem in root.iter():
            tag_without_ns = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag_without_ns == 'rect':
                if elem.attrib.get('width') == '1440':
                    elem.attrib['width'] = '960'
                    
        # Find all child elements of root <svg> that have x attribute. If x >= 950, remove it.
        children_to_remove = []
        for child in root:
            x_val = child.attrib.get('x')
            if x_val is not None:
                try:
                    if float(x_val) >= 950:
                        children_to_remove.append(child)
                except ValueError:
                    pass
                    
        for child in children_to_remove:
            root.remove(child)
            
        modified_svg_xml = ET.tostring(root, encoding='utf-8').decode('utf-8')
        
        # Wrap modified SVG in the required HTML template
        html_content = f"""<!DOCTYPE html>
<html>
<head>
<style>
  html, body {{ margin: 0; padding: 0; overflow: hidden; background-color: #f8fafc; }}
  svg {{ display: block; width: 960px; height: {height}px; }}
</style>
</head>
<body>
{modified_svg_xml}
</body>
</html>
"""
        
        # Write to a temp HTML file
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
            f.write(html_content)
            temp_html_path = f.name
            
        file_url = "file:///" + os.path.abspath(temp_html_path).replace("\\", "/")
        
        # Create a unique user data dir to avoid profile conflicts
        user_data_dir = tempfile.mkdtemp(prefix="chrome_user_data_")
        
        cmd = [
            browser_path,
            "--headless",
            "--disable-gpu",
            f"--user-data-dir={user_data_dir}",
            f"--screenshot={png_path}",
            f"--window-size=960,{height}",
            file_url
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Clean up temp files
        try:
            os.remove(temp_html_path)
            shutil.rmtree(user_data_dir, ignore_errors=True)
        except Exception:
            pass
            
        if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
            return True, basename
        else:
            return False, f"{basename}: file not created or size is 0. Cmd exit code: {result.returncode}. Stderr: {result.stderr}"
            
    except Exception as e:
        return False, f"{os.path.basename(svg_path)}: {str(e)}"

def main():
    start_time = time.time()
    try:
        browser_path = get_browser_path()
        print(f"Using browser for rendering: {browser_path}")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    # Discover SVG files
    svg_pattern = os.path.join(TASKS_DIR, "*.svg")
    svg_files = glob.glob(svg_pattern)
    
    # Filter 0001_ to 0800_
    target_files = []
    for f in svg_files:
        basename = os.path.basename(f)
        if len(basename) >= 4 and basename[:4].isdigit():
            num = int(basename[:4])
            if 1 <= num <= 800:
                target_files.append(f)
                
    target_files = sorted(target_files)
    total_files = len(target_files)
    print(f"Found {total_files} target SVG files to convert (0001 to 0800).")
    
    if total_files == 0:
        print("No target files found. Exiting.")
        sys.exit(0)
        
    print("Starting conversion using ThreadPoolExecutor with 8 workers...")
    success_count = 0
    failures = []
    
    # Use ThreadPoolExecutor to run tasks in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        futures = {executor.submit(process_file, f, browser_path): f for f in target_files}
        
        for i, future in enumerate(futures):
            success, msg = future.result()
            if success:
                success_count += 1
            else:
                failures.append(msg)
                
            if (i + 1) % 50 == 0 or (i + 1) == total_files:
                elapsed = time.time() - start_time
                print(f"Processed {i + 1}/{total_files} files... Successes: {success_count}, Failures: {len(failures)}. Elapsed: {elapsed:.1f}s")
                
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nFinished conversion in {duration:.2f} seconds.")
    print(f"Total Successes: {success_count}/{total_files}")
    
    if failures:
        print(f"Total Failures: {len(failures)}")
        for f_msg in failures[:10]:
            print(f"  - {f_msg}")
        if len(failures) > 10:
            print(f"  - ... and {len(failures) - 10} more failures.")
            
    # Perform automated verification
    print("\n--- Running Verification ---")
    verified_count = 0
    missing_files = []
    zero_size_files = []
    
    for f in target_files:
        basename = os.path.basename(f)
        png_name = basename.replace(".svg", ".png")
        png_path = os.path.join(TASKS_DIR, png_name)
        
        if not os.path.exists(png_path):
            missing_files.append(png_name)
        elif os.path.getsize(png_path) == 0:
            zero_size_files.append(png_name)
        else:
            verified_count += 1
            
    print(f"Verification Results:")
    print(f"  - Expected PNGs: {total_files}")
    print(f"  - Successfully verified (exists & >0 bytes): {verified_count}")
    
    if missing_files:
        print(f"  - Missing PNGs ({len(missing_files)}): {missing_files[:10]}")
    if zero_size_files:
        print(f"  - Zero-size PNGs ({len(zero_size_files)}): {zero_size_files[:10]}")
        
    if verified_count == total_files:
        print("Verification SUCCESS! All files converted perfectly.")
        sys.exit(0)
    else:
        print("Verification FAILED. Some files are missing or zero-sized.")
        sys.exit(1)

if __name__ == "__main__":
    main()
