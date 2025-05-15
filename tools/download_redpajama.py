import os
import time
import requests
import multiprocessing
from urllib.parse import urlparse
from tqdm import tqdm
from multiprocessing import Pool, Lock

# Try importing zstandard, provide instructions if missing
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    print("INFO: 'zstandard' library not found. To handle .jsonl.zst files, please install it: pip install zstandard")
    ZSTD_AVAILABLE = False

# Global lock for synchronized print output
lock = None

def init_child(lock_):
    global lock
    lock = lock_

def download_url(url):
    global lock
    max_retries = 3
    temp_path = None  # To track the temporary file path

    # Determine the final download location and if it's zstd compressed
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.split('/')
    # Assuming structure like /redpajama-data-1T/v1.0.0/<subset>/<file>.jsonl or .jsonl.zst
    # Adjust indices if URL structure is different
    if len(path_parts) > 3:
         relevant_parts = path_parts[3:]
    else:
         # Handle unexpected URL paths gracefully
         relevant_parts = path_parts[1:] if len(path_parts) > 1 else ['downloaded_file']
         if not relevant_parts[-1]: # if path ends in '/', use a default name
             relevant_parts[-1] = 'index' if len(relevant_parts) == 1 else path_parts[-2]


    dload_loc_original = os.path.join(*relevant_parts)
    is_zst = dload_loc_original.endswith('.jsonl.zst')
    
    # Define the final path (decompressed if necessary)
    if is_zst:
        final_dload_loc = dload_loc_original[:-4] # Remove '.zst'
    else:
        final_dload_loc = dload_loc_original
        
    # Create directory for the *final* file path
    os.makedirs(os.path.dirname(final_dload_loc), exist_ok=True)

    # Skip .zst if library is not available
    if is_zst and not ZSTD_AVAILABLE:
        with lock:
            print(f"Skipping {url}: zstandard library not installed.")
        return False # Treat as failure for summary

    for attempt in range(max_retries):
        try:
            # Get file info (Content-Length refers to compressed size for .zst)
            head_response = requests.head(url, timeout=30, allow_redirects=True)
            head_response.raise_for_status()
            content_length = int(head_response.headers.get('Content-Length', 0))

            if content_length == 0:
                with lock:
                    print(f"Skipping {url}: Content-Length is zero.")
                return False # Or True if zero-length is considered 'handled'

            # Calculate 1% download size (based on original/compressed size)
            # Note: 1% of compressed data might decompress to more or less than 1% of lines
            one_percent_size = (content_length + 99) // 100
            
            # Ensure we download at least a minimal amount for small files,
            # especially for .zst where headers are crucial. Maybe 1KB minimum?
            min_download_bytes = 1024
            download_target_size = max(one_percent_size, min_download_bytes if is_zst else 1)
            # But don't download more than the total content length
            download_target_size = min(download_target_size, content_length)


            # Unique temporary file name for this attempt
            temp_path = final_dload_loc + f'.partial.{os.getpid()}.{attempt}'

            # Clean up potentially leftover temp file from previous runs/attempts
            if os.path.exists(temp_path):
                os.remove(temp_path)

            # Execute download of the partial content
            downloaded = 0
            with requests.get(url, stream=True, timeout=60, allow_redirects=True) as response:
                response.raise_for_status()
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if downloaded >= download_target_size:
                            break
                        remaining = download_target_size - downloaded
                        if len(chunk) > remaining:
                            chunk = chunk[:remaining]
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= download_target_size:
                            break # Ensure we break immediately after reaching target

            # Verify some data was downloaded
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                 # This might happen if the file is smaller than min_download_bytes
                 # or if the download failed silently.
                 if downloaded == 0 and content_length > 0 :
                    raise RuntimeError(f"Downloaded partial file {temp_path} is empty despite Content-Length {content_length}")
                 # else: If content_length was 0 or very small, empty file might be expected.

            # --- Process the downloaded partial file ---
            raw_content = b''
            try:
                if is_zst:
                    # Attempt to decompress the partial .zst file
                    dctx = zstd.ZstdDecompressor()
                    try:
                        with open(temp_path, 'rb') as f_in:
                            # stream_reader can handle incomplete streams better
                            reader = dctx.stream_reader(f_in)
                            raw_content = reader.read()
                    except zstd.ZstdError as e:
                        # Decompression likely failed due to partial nature.
                        # We might have gotten *some* data in raw_content before the error.
                        with lock:
                            print(f"Warning: Decompression error for partial {url} (likely truncated stream): {e}. Processing any data extracted.")
                        # raw_content will contain whatever was read before the error
                else:
                    # Handle regular .jsonl files
                    with open(temp_path, 'rb') as f:
                        raw_content = f.read()

            except Exception as proc_e:
                 # Catch errors during decompression or file reading
                 with lock:
                     print(f"Error processing temporary file {temp_path} for {url}: {proc_e}")
                 # Clean up before raising or continuing retry loop
                 if os.path.exists(temp_path):
                     os.remove(temp_path)
                 raise # Re-raise the exception to trigger retry or failure

            # Process the (potentially decompressed) content to find complete lines
            processed_content = b''
            if raw_content:
                # Find the last newline character
                last_newline = raw_content.rfind(b'\n')
                if last_newline != -1:
                    # Keep everything up to and including the last newline
                    processed_content = raw_content[:last_newline + 1]
                # else: No newline found, discard the incomplete fragment by leaving processed_content empty

            # Write the processed content to the *final* file location
            lines_extracted = processed_content.count(b'\n')
            if processed_content:
                with open(final_dload_loc, 'wb') as f:
                    f.write(processed_content)
            else:
                # If no complete lines were found, we might still consider the download attempt "successful"
                # but we won't create an empty file. We'll log it.
                with lock:
                     print(f"Info: No complete JSON lines extracted from partial download of {url} (Downloaded {downloaded} bytes).")


            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            with lock:
                 status_msg = f"Processed partial {url} -> {os.path.basename(final_dload_loc)}"
                 status_msg += f" ({lines_extracted} lines extracted from {downloaded} bytes downloaded)."
                 print(status_msg)
            return True # Indicate success for this URL

        except requests.exceptions.RequestException as e:
            retry_msg = f"Attempt {attempt + 1}/{max_retries} failed for {url}: {e}"
            # Clean up temp file before retry/failure
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

            if attempt == max_retries - 1:
                with lock:
                    print(f"Failed to download {url} after {max_retries} attempts: {e}")
                return False
            else:
                 wait_time = 5 * (2 ** attempt) # Exponential backoff
                 with lock:
                     print(f"{retry_msg}. Retrying in {wait_time} seconds...")
                 time.sleep(wait_time)
        except Exception as e:
             # Catch other unexpected errors during the process
             if temp_path and os.path.exists(temp_path):
                 os.remove(temp_path)
             with lock:
                 print(f"Unexpected error processing {url} on attempt {attempt + 1}: {e}")
             # Decide if retry makes sense or fail immediately
             if attempt == max_retries - 1:
                return False
             else:
                 wait_time = 5 * (2 ** attempt)
                 time.sleep(wait_time) # Retry after backoff

    # Should not be reached if retries are handled correctly, but as a fallback:
    return False


if __name__ == '__main__':
    # Ensure the base directory for downloads exists (optional, handled in func too)
    # os.makedirs('download_output', exist_ok=True) # Example base dir

    # Download urls.txt if it doesn't exist
    urls_txt_url = 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt'
    urls_file_path = 'urls.txt'
    if not os.path.exists(urls_file_path):
        print(f"Downloading URL list from {urls_txt_url}...")
        try:
            response = requests.get(urls_txt_url)
            response.raise_for_status()
            with open(urls_file_path, 'wb') as f:
                f.write(response.content)
            print("URL list downloaded.")
        except requests.exceptions.RequestException as e:
            print(f"Fatal: Could not download {urls_file_path}: {e}")
            exit(1) # Exit if we can't get the URLs

    # Read URL list
    try:
        with open(urls_file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and (line.strip().endswith('.jsonl') or line.strip().endswith('.jsonl.zst'))]
        print(f"Found {len(urls)} .jsonl or .jsonl.zst URLs in {urls_file_path}")
    except FileNotFoundError:
        print(f"Fatal: {urls_file_path} not found.")
        exit(1)


    # Create process pool
    # manager = multiprocessing.Manager() # Manager not needed if lock is global in module
    lock = Lock()
    num_processes = 16 # Adjust as needed for your system

    print(f"Starting download process with {num_processes} workers...")
    with Pool(processes=num_processes, initializer=init_child, initargs=(lock,)) as pool:
        # Use imap_unordered for potentially better throughput as results come in any order
        results_iterator = pool.imap_unordered(download_url, urls)
        
        # Wrap the iterator with tqdm for progress bar
        results = list(tqdm(
            results_iterator,
            total=len(urls),
            desc="Processing files",
            unit="file"
        ))

    # Tally results
    success_count = sum(1 for r in results if r is True) # Count True results
    failure_count = len(urls) - success_count
    print("\n--- Download Summary ---")
    print(f"Total URLs processed: {len(urls)}")
    print(f"Successful attempts: {success_count}")
    print(f"Failed attempts:    {failure_count}")
    print("------------------------")
    