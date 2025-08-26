#!/usr/bin/env python3

# VidCacheFS - FUSE cache filesystem for media pre-warm
# Version: 0.1.0 (2025-08-23)
# Purpose: Near-instant Plex playback in environments with spundown HDDs by caching critical portions of video files + full subtitles on (NVME) SSD.
# See README for detailed features, limitations, and usage.
# Adjust the embedded configuration to your setup.
__version__ = "0.1.0"

import os
import sys
import errno
import time
import json
import glob
import threading
import argparse
import logging
import shutil
import multiprocessing
import stat  # added for permission bit checks
from collections import OrderedDict
from fuse import FUSE, Operations, LoggingMixIn
import subprocess

try:
    import inotify.adapters
    INOTIFY_AVAILABLE = True
except ImportError:
    INOTIFY_AVAILABLE = False
    logging.warning("inotify module not available. File watching disabled. Run: pip install inotify")
logging.basicConfig(level = logging.INFO,format='%(created).6f [%(levelname)s] %(message)s')

# ---------- Default embedded configuration ----------
DEFAULT_CONFIG = {
  "GLOBAL": {
    "VERBOSE": True, # Enable verbose logging
    "MAX_CACHE_SIZE_BYTES": 1850 * 1024 * 1024 * 1024, # Total cache size available on the disk/folder
    "READ_ONLY": False, # If true, mount read-only (no cache writes)
    "CHECK_FILE_MODIFICATIONS": False # Use when inotify is not available and you want to detect file modifications on the backing drives
  },
"MOUNTS": [
    {
      "BACKING_DIR": "/mnt/user/video/Movies",
      "MOUNT_POINT": "/mnt/ssd_cache/Movies",
      "CACHE_DIR": "/mnt/videocache/Movies",
      "MAX_FILES": 1500, # Maximum number of media files to cache. Cache Limit set to MAX_FILES*(HEAD_BYTES+TAIL+BYTES)
      "HEAD_BYTES": 75 * 1024 * 1024,
      "TAIL_BYTES": 1 * 1024 * 1024,
      "SCHEDULES": [
        {
          "path": "/mnt/user/video/Movies",
          "scan_interval_seconds": 14400,
          "pattern": "*.{mkv,mp4,avi,mov,m4v,mpg,mpeg,wmv,flv,webm}"
        }
      ],
      "OPEN_SPINUP_TTL": 5
    },
    {
      "BACKING_DIR": "/mnt/user/video/Series",
      "MOUNT_POINT": "/mnt/ssd_cache/Series",
      "CACHE_DIR": "/mnt/videocache/Series",
      "MAX_FILES": 25000,
      "HEAD_BYTES": 60 * 1024 * 1024,
      "TAIL_BYTES": 1 * 1024 * 1024,
      "SCHEDULES": [
        {
          "path": "/mnt/user/video/Series",
          "scan_interval_seconds": 14400,
          "pattern": "*.{mkv,mp4,avi,mov,m4v,mpg,mpeg,wmv,flv,webm}"
        }
      ],
      "OPEN_SPINUP_TTL": 5
    }
  ],
  "SUBTITLE_EXTENSIONS": [".srt", ".vtt", ".sub", ".ass", ".ssa", ".idx", ".smi"], #   Subtitle file extensions to fully cache
  "VIDEO_EXTENSIONS": [".mkv", ".mp4", ".avi", ".mov", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".webm"] # Video file extensions to partially cache
}

# Define standard locations to look for config files
CONFIG_PATHS = [
    "/mnt/user/appdata/ssd_cache_config.json",
    "config.json"
]

# ---------- Utility functions ----------
def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def safe_copy_range(src_path, dst_path, length, offset=0, bufsize=4 * 1024 * 1024):
    """Copy `length` bytes from src_path starting at offset into dst_path."""
    ensure_parent_dir(dst_path)
    with open(src_path, 'rb') as s, open(dst_path, 'wb') as d:
        s.seek(offset)
        remaining = length
        while remaining > 0:
            toread = min(bufsize, remaining)
            chunk = s.read(toread)
            if not chunk:
                break
            d.write(chunk)
            remaining -= len(chunk)


def safe_copy_file(src_path, dst_path):
    """Copy an entire file from src_path to dst_path."""
    ensure_parent_dir(dst_path)
    shutil.copy2(src_path, dst_path)

class MyLoggingMixIn(LoggingMixIn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mylog = logging.getLogger("fuse_custom")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.mylog.addHandler(handler)
        self.mylog.setLevel(logging.INFO)
        
# ---------- Cache Manager (LRU) ----------
class CacheManager:
    def __init__(self, cache_dir, max_cache_bytes, verbose=False):
        self.cache_dir = cache_dir
        self.max_cache_bytes = max_cache_bytes
        self.verbose = verbose
        # mapping from cache_path -> (size, last_access_time)
        self.lock = threading.RLock()
        self._load_existing_cache()

    def _load_existing_cache(self):
        # Build LRU state from cache directory
        self.lru = OrderedDict()
        total = 0
        for root, dirs, files in os.walk(self.cache_dir):
            for f in files:
                if f.endswith('.head') or f.endswith('.tail') or f.endswith('.full'):
                    full = os.path.join(root, f)
                    try:
                        sz = os.path.getsize(full)
                    except OSError:
                        continue
                    total += sz
                    atime = os.path.getatime(full)
                    self.lru[full] = (sz, atime)
        # Order by access time ascending (least recently used first)
        self.lru = OrderedDict(sorted(self.lru.items(), key=lambda kv: kv[1][1]))
        self.current_cache_bytes = total
        if self.verbose:
            logging.info(f"Loaded cache state: {len(self.lru)} items, {self.current_cache_bytes} bytes")

    def touch(self, cache_path):
        with self.lock:
            if cache_path in self.lru:
                sz, _ = self.lru.pop(cache_path)
                at = time.time()
                self.lru[cache_path] = (sz, at)
                try:
                    os.utime(cache_path, (at, os.path.getmtime(cache_path)))
                except Exception:
                    pass

    def add(self, cache_path):
        # Add a new cache file into LRU
        with self.lock:
            try:
                sz = os.path.getsize(cache_path)
            except OSError:
                return
            at = time.time()
            if cache_path in self.lru:
                self.lru.pop(cache_path)
            self.lru[cache_path] = (sz, at)
            self.current_cache_bytes += sz
            if self.verbose:
                logging.info(f"Added cache item {cache_path} ({sz} bytes). Total {self.current_cache_bytes}")
            self._evict_if_needed()

    def _evict_if_needed(self):
        with self.lock:
            while self.current_cache_bytes > self.max_cache_bytes and self.lru:
                # pop oldest
                oldest, (sz, at) = self.lru.popitem(last=False)
                try:
                    os.remove(oldest)
                    # Also remove .meta if present
                    if oldest.endswith('.head'):
                        meta = oldest[:-5] + '.meta'
                    elif oldest.endswith('.tail'):
                        meta = oldest[:-5] + '.meta'
                    elif oldest.endswith('.full'):
                        meta = oldest[:-5] + '.meta'
                    else:
                        meta = None
                    if meta and os.path.exists(meta):
                        os.remove(meta)
                except Exception as e:
                    logging.exception(f"Failed to remove cache item {oldest}: {e}")
                self.current_cache_bytes -= sz
                if self.verbose:
                    logging.info(f"Evicted {oldest} ({sz}) to free space. New total {self.current_cache_bytes}")

    def get_path_for(self, relpath, kind):
        # kind in ('head','tail','meta','full')
        safe_rel = relpath.lstrip('/')
        return os.path.join(self.cache_dir, safe_rel + f'.{kind}')
        
    # --- Added: one-time reconciliation on startup ---
    def reconcile_startup(self, head_bytes, tail_bytes, max_files):
        """Executed once after startup to:
        1. Remove head/tail cache pieces whose size no longer matches configured head/tail bytes.
        2. If cached file groups (.meta present) exceed max_files, prune oldest groups (LRU based on atime of their parts).
        Only files with a .meta are considered valid groups. Subtitles (.full) are not resized.
        """
        with self.lock:
            if self.verbose:
                logging.info(f"Starting cache reconciliation: head_bytes={head_bytes}, tail_bytes={tail_bytes}, max_files={max_files}")
            # Build group map: base_path (without extension) -> dict(parts)
            groups = {}
            for root, _, files in os.walk(self.cache_dir):
                for name in files:
                    if not (name.endswith('.meta') or name.endswith('.head') or name.endswith('.tail') or name.endswith('.full')):
                        continue
                    full = os.path.join(root, name)
                    base, _ = os.path.splitext(full)
                    g = groups.setdefault(base, {'meta': None, 'head': None, 'tail': None, 'full': None})
                    if name.endswith('.meta'):
                        g['meta'] = full
                    elif name.endswith('.head'):
                        g['head'] = full
                    elif name.endswith('.tail'):
                        g['tail'] = full
                    elif name.endswith('.full'):
                        g['full'] = full
            # Keep only groups having meta
            groups = {b: p for b, p in groups.items() if p['meta']}
            # Pass 1: resize validation (delete mismatched head/tail)
            for base, parts in groups.items():
                try:
                    with open(parts['meta'], 'r') as f:
                        meta = json.load(f)
                    fsize = meta.get('size')
                    if fsize is None:
                        continue
                    # Expected sizes
                    expected_head_size = min(head_bytes, fsize) if head_bytes > 0 else 0
                    expected_tail_size = min(tail_bytes, fsize) if tail_bytes > 0 else 0
                    # head check
                    hp = parts.get('head')
                    if hp and os.path.exists(hp):
                        try:
                            if os.path.getsize(hp) != expected_head_size:
                                os.remove(hp)
                                if hp in self.lru:
                                    sz,_ = self.lru.pop(hp)
                                    self.current_cache_bytes -= sz
                                if self.verbose:
                                    logging.info(f"Removed mismatched head cache: {hp}")
                        except Exception:
                            pass
                    # tail check
                    tp = parts.get('tail')
                    if tp and os.path.exists(tp):
                        try:
                            if os.path.getsize(tp) != expected_tail_size:
                                os.remove(tp)
                                if tp in self.lru:
                                    sz,_ = self.lru.pop(tp)
                                    self.current_cache_bytes -= sz
                                if self.verbose:
                                    logging.info(f"Removed mismatched tail cache: {tp}")
                        except Exception:
                            pass
                except Exception:
                    continue
            # Pass 2: enforce max_files (count of groups)
            group_count = len(groups)
            if group_count > max_files:
                # Build age list using min atime among existing part files
                age_list = []  # (atime, base, parts)
                for base, parts in groups.items():
                    at_list = []
                    for p in (parts.get('head'), parts.get('tail'), parts.get('full')):
                        if p and os.path.exists(p):
                            try:
                                at_list.append(os.path.getatime(p))
                            except Exception:
                                pass
                    if not at_list:  # fallback to meta
                        try:
                            at_list.append(os.path.getatime(parts['meta']))
                        except Exception:
                            at_list.append(time.time())
                    age_list.append((min(at_list), base, parts))
                age_list.sort(key=lambda x: x[0])  # oldest first
                to_remove = group_count - max_files
                removed = 0
                for _, base, parts in age_list:
                    if removed >= to_remove:
                        break
                    for p in (parts.get('head'), parts.get('tail'), parts.get('full'), parts.get('meta')):
                        if p and os.path.exists(p):
                            try:
                                size = os.path.getsize(p)
                            except Exception:
                                size = 0
                            try:
                                os.remove(p)
                                if p in self.lru:
                                    sz,_ = self.lru.pop(p)
                                    self.current_cache_bytes -= sz
                                else:
                                    self.current_cache_bytes -= size
                            except Exception:
                                pass
                    removed += 1
                if self.verbose:
                    logging.info(f"Startup prune: removed {removed} old cached groups (now {group_count-removed}/{max_files}).")


# ---------- Scheduler: populates cache on interval ----------
class Scheduler(threading.Thread):
    def __init__(self, schedules, cache_manager, head_bytes, tail_bytes, backing_root, 
                 subtitle_extensions, video_extensions, verbose=False, use_cache_db=False,
                 check_file_modifications=False):
        super().__init__(daemon=True)
        self.schedules = schedules
        self.cache_manager = cache_manager
        self.head_bytes = head_bytes
        self.tail_bytes = tail_bytes
        self.backing_root = backing_root
        self.subtitle_extensions = subtitle_extensions
        self.video_extensions = video_extensions
        self.verbose = verbose
        self._stop = threading.Event()
        self.cache = cache_manager  # Alias for readability
        self.use_cache_db = use_cache_db
        self.last_db_check = 0
        self.db_check_interval = 60  # Check database every 60 seconds
        self.cache_db_path = os.path.join(self.cache_manager.cache_dir, ".cache_db.json")
        
        # Add file modification check option
        self.check_file_modifications = check_file_modifications
        if self.check_file_modifications and self.verbose:
            logging.info("File modification checking enabled - may cause drives to spin up during scheduled scans")
        
        # Existing code for buffer initialization
        self.cache_buffer = OrderedDict()  # Maps path -> (mtime, size, relpath)
        self.buffer_size_bytes = 0
        self.max_buffer_size = self.cache_manager.max_cache_bytes  # Use same limit as cache

    def run(self):
        logging.info(f"Cache scheduler started for {self.backing_root}")
        # Schedule is map of path -> interval
        last_scan = {}
        
        while not self._stop.is_set():
            now = time.time()
            
            # First check cache database for pending files (this doesn't cause drive spin-up)
            if self.use_cache_db and now - self.last_db_check >= self.db_check_interval:
                self.last_db_check = now
                try:
                    self._check_cache_database()
                except Exception:
                    logging.exception("Cache database check failed")
            
            # Then do normal scheduled scans
            for sched in self.schedules:
                path = sched['path']
                interval = sched['scan_interval_seconds']
                pattern = sched['pattern']
                # Check if we should run this schedule
                if path not in last_scan or now - last_scan[path] >= interval:
                    last_scan[path] = now
                    try:
                        if self.verbose:
                            logging.info(f"Running cache scan for {path} with pattern {pattern}")
                        self._scan_and_buffer(path, pattern)
                    except Exception as e:
                        logging.exception(f"Scheduler scan failed: {e}")
                    
                    # After scan, process the buffer to cache files
                    self._process_buffer()
            
            # Sleep a small amount so we can be responsive to stop
            self._stop.wait(5)

    def _scan_and_buffer(self, path, pattern):
        """Scan for files matching pattern and add them to buffer without immediately caching."""
        # For patterns with multiple extensions like *.{mkv,mp4}, pass them directly
        # to _scan_files_into_buffer which now handles brace expansion internally
        self._scan_files_into_buffer(path, pattern)
        
        # Also scan for subtitle files - each extension separately for clarity
        for ext in self.subtitle_extensions:
            self._scan_files_into_buffer(path, f"*{ext}")

    def _scan_files_into_buffer(self, path, pattern):
        """Scan files by pattern and add to buffer with LRU eviction."""
        try:
            # For large directories, using glob recursively can hit system limits
            # Use os.walk instead which is more efficient for large directory structures
            total_files_found = 0
            
            if '{' in pattern and '}' in pattern:
                # Handle brace expansion patterns like "*.{mkv,mp4}"
                base, extensions = pattern.split('{', 1)
                extensions = extensions.split('}', 1)[0]
                extensions_list = [ext.strip() for ext in extensions.split(',')]
                # Create a list of patterns to match
                patterns = [base + ext for ext in extensions_list]
            else:
                patterns = [pattern]
                
            # Walk the directory tree manually
            for root, _, files in os.walk(path):
                for filename in files:
                    # Check each filename against our patterns
                    for pat in patterns:
                        if self._matches_simple_pattern(filename, pat):
                            total_files_found += 1
                            full_path = os.path.join(root, filename)
                            
                            # Check if this is a file type we care about
                            _, ext = os.path.splitext(filename.lower())
                            if ext not in self.subtitle_extensions and ext not in self.video_extensions:
                                continue
                                
                            try:
                                # Get file stats
                                st = os.stat(full_path)
                                mtime = st.st_mtime
                                size = st.st_size
                                
                                # Calculate how much space this will use in cache
                                if ext in self.subtitle_extensions:
                                    cache_size = size + 1024  # file + metadata
                                else:
                                    cache_size = min(self.head_bytes, size) + min(self.tail_bytes, size) + 1024
                                
                                # Get relative path
                                try:
                                    rel = os.path.relpath(full_path, self.backing_root)
                                except ValueError:
                                    # Not under backing_root
                                    rel = os.path.basename(full_path)
                                
                                # Add to buffer, possibly evicting older entries
                                self._add_to_buffer(full_path, mtime, cache_size, rel)
                                
                                # Break out of pattern loop since we found a match
                                break
                                
                            except (FileNotFoundError, PermissionError):
                                # Skip files we can't access
                                continue
                            except Exception as e:
                                if self.verbose:
                                    logging.warning(f"Error processing file {full_path}: {e}")
            
            if self.verbose:
                logging.info(f"Found {total_files_found} files matching pattern(s) {patterns} in {path}")
                logging.info(f"Buffer now contains {len(self.cache_buffer)} files ({self.buffer_size_bytes/1024/1024:.2f} MB)")
                
        except Exception as e:
            logging.warning(f"Error scanning path {path} with pattern {pattern}: {e}")
            
    def _matches_simple_pattern(self, filename, pattern):
        """Simple pattern matching for filenames (faster than regex for large datasets).
        
        Supports basic wildcard patterns like "*.mp4" or "video.*"
        """
        if pattern.startswith('*.'):
            # Most common case - extension matching
            return filename.lower().endswith(pattern[1:].lower())
        elif pattern == '*':
            # Match everything
            return True
        elif '*' not in pattern:
            # Exact match
            return filename.lower() == pattern.lower()
        elif pattern.endswith('*'):
            # Starts with pattern
            prefix = pattern[:-1]
            return filename.lower().startswith(prefix.lower())
        elif pattern.startswith('*'):
            # Ends with pattern
            suffix = pattern[1:]
            return filename.lower().endswith(suffix.lower())
        else:
            # More complex pattern - parts before and after *
            parts = pattern.split('*', 1)
            return (filename.lower().startswith(parts[0].lower()) and 
                    filename.lower().endswith(parts[1].lower()))

    def _add_to_buffer(self, filepath, mtime, cache_size, relpath):
        """Add a file to the buffer, evicting older files if needed."""
        # If file is already in buffer, update its timestamp (make it most recent)
        if filepath in self.cache_buffer:
            _, old_size, _ = self.cache_buffer.pop(filepath)
            self.buffer_size_bytes -= old_size
        
        # Add the new file to the buffer (at the end, most recent)
        self.cache_buffer[filepath] = (mtime, cache_size, relpath)
        self.buffer_size_bytes += cache_size
        
        # Evict older files if we exceed the buffer size limit
        while self.buffer_size_bytes > self.max_buffer_size and self.cache_buffer:
            # Remove the oldest entry (first in the OrderedDict)
            oldest_path, (_, oldest_size, _) = next(iter(self.cache_buffer.items()))
            self.cache_buffer.pop(oldest_path)
            self.buffer_size_bytes -= oldest_size
            
            if self.verbose and len(self.cache_buffer) % 100 == 0:
                logging.debug(f"Buffer: {len(self.cache_buffer)} files, {self.buffer_size_bytes/1024/1024:.2f} MB")

    def _process_buffer(self):
        """Process all files in the buffer and cache them using a single worker."""
        if not self.cache_buffer:
            return
        
        count = 0
        cache_size = 0
        buffer_size = len(self.cache_buffer)
        
        if self.verbose:
            logging.info(f"Processing cache buffer with {buffer_size} files")
        
        try:
            # Process files from buffer (most recently modified first)
            progress_interval = max(1, int(buffer_size * 0.1))  # Report progress at 10% intervals
            
            for i, (filepath, (mtime, size_estimate, relpath)) in enumerate(list(self.cache_buffer.items())):
                try:
                    self._ensure_cache_for(filepath, relpath)
                    count += 1
                    cache_size += size_estimate
                    
                    # Report progress more frequently for large buffer sizes
                    if self.verbose and (i+1) % progress_interval == 0:
                        percent = 100.0 * (i+1) / buffer_size
                        logging.info(f"Cached {count}/{buffer_size} files ({percent:.1f}%)")
                        
                    # Check if we should stop
                    if self._stop.is_set():
                        break
                        
                except Exception as e:
                    logging.warning(f"Error caching file {filepath}: {e}")
            
            # Clear the buffer after processing
            self.cache_buffer.clear()
            self.buffer_size_bytes = 0
            
            if self.verbose:
                logging.info(f"Completed caching {count} files, ~{cache_size/1024/1024:.2f} MB")
                
        except Exception as e:
            logging.exception(f"Unexpected error in _process_buffer: {e}")


    def stop(self):
        self._stop.set()
        self.join(timeout=2)

    def _scan_and_cache(self, path, pattern):
        # Handle patterns with multiple extensions like *.{mkv,mp4}
        if '{' in pattern and '}' in pattern:
            # Extract the extension part from the pattern
            base, extensions = pattern.split('{', 1)
            extensions = extensions.split('}', 1)[0]
            extensions = extensions.split(',')
            for ext in extensions:
                self._scan_files_by_pattern(path, base + ext.strip())
        else:
            self._scan_files_by_pattern(path, pattern)
        
        # Also scan for subtitle files
        for ext in self.subtitle_extensions:
            self._scan_files_by_pattern(path, f"*{ext}")

    def _scan_files_by_pattern(self, path, pattern):
        globp = os.path.join(path, '**', pattern)
        for file in glob.glob(globp, recursive=True):
            if not os.path.isfile(file):
                continue
            try:
                rel = os.path.relpath(file, self.backing_root)
            except ValueError:
                # Not under backing_root
                rel = os.path.basename(file)
            self._ensure_cache_for(file, rel)

    def _ensure_cache_for(self, fullpath, relpath):
        # Check if this is a subtitle file
        _, ext = os.path.splitext(fullpath.lower())
        is_subtitle = ext in self.subtitle_extensions
        is_video = ext in self.video_extensions
        
        # For non-subtitle/video files, we don't cache
        if not is_subtitle and not is_video:
            return
            
        # Check meta to see if cache valid
        if is_subtitle:
            # For subtitle files, cache the entire file
            full_path = self.cache_manager.get_path_for(relpath, 'full')
            meta_path = self.cache_manager.get_path_for(relpath, 'meta')
            head_path = None
            tail_path = None
        else:
            # For video files, cache head and tail portions
            head_path = self.cache_manager.get_path_for(relpath, 'head')
            tail_path = self.cache_manager.get_path_for(relpath, 'tail')
            meta_path = self.cache_manager.get_path_for(relpath, 'meta')
            full_path = None

        need_head = not is_subtitle
        need_tail = not is_subtitle
        need_full = is_subtitle
        need_meta = True  # Always assume we need metadata initially
        
        # First get file stats for metadata
        try:
            st = os.stat(fullpath)
            metadata = {
                'size': st.st_size,
                'mtime': st.st_mtime,
                'atime': st.st_atime,
                'ctime': st.st_ctime,
                'mode': st.st_mode,
                'uid': st.st_uid,
                'gid': st.st_gid,
                'nlink': st.st_nlink
            }
        except Exception as e:
            if self.verbose:
                logging.warning(f"Failed to get stats for {relpath}: {e}")
            return  # Cannot continue without file stats
        
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    old = json.load(f)
                
                # Check if file has changed when file modification checking is enabled
                if self.check_file_modifications:
                    # We already have current stats, so just compare
                    if (old.get('size') != metadata['size'] or old.get('mtime') != metadata['mtime']):
                        if self.verbose:
                            logging.info(f"File changed (size/mtime): {relpath}")
                            if old.get('size') != metadata['size']:
                                logging.info(f"  Size changed: {old.get('size')} -> {metadata['size']}")
                            if old.get('mtime') != metadata['mtime']:
                                logging.info(f"  Mtime changed: {old.get('mtime')} -> {metadata['mtime']}")
                        # Force recaching by keeping need_* as True
                        need_head = not is_subtitle
                        need_tail = not is_subtitle
                        need_full = is_subtitle
                        need_meta = True
                    else:
                        # File is unchanged, check if cache exists
                        need_meta = False  # We already have valid metadata
                        if is_subtitle:
                            if os.path.exists(full_path) and os.path.getsize(full_path) >= old.get('size', 0):
                                need_full = False
                        else:
                            if os.path.exists(head_path) and os.path.getsize(head_path) >= min(self.head_bytes, old.get('size', 0)):
                                need_head = False
                            if os.path.exists(tail_path) and os.path.getsize(tail_path) >= min(self.tail_bytes, old.get('size', 0)):
                                need_tail = False
                else:
                    # Standard check without comparing file modifications
                    need_meta = False  # Existing metadata is fine
                    if old.get('size') == None or old.get('mtime') == None:
                        # Metadata is incomplete, force recache
                        need_head = not is_subtitle
                        need_tail = not is_subtitle
                        need_full = is_subtitle
                        need_meta = True
                    else:
                        # Just check if cache files exist
                        if is_subtitle:
                            if os.path.exists(full_path) and os.path.getsize(full_path) >= old.get('size', 0):
                                need_full = False
                        else:
                            if os.path.exists(head_path) and os.path.getsize(head_path) >= min(self.head_bytes, old.get('size', 0)):
                                need_head = False
                            if os.path.exists(tail_path) and os.path.getsize(tail_path) >= min(self.tail_bytes, old.get('size', 0)):
                                need_tail = False
            except Exception as e:
                if self.verbose:
                    logging.warning(f"Failed to read metadata for {relpath}: {e}")
                # If we can't read metadata, force recache
                need_head = not is_subtitle
                need_tail = not is_subtitle
                need_full = is_subtitle
                need_meta = True

        # Write metadata if needed
        if need_meta:
            ensure_parent_dir(meta_path)
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
        
        # If file changed or no meta, we will write new cache parts
        if is_subtitle and need_full:
            try:
                if self.verbose: 
                    logging.info(f"Caching full subtitle file {relpath}")
                ensure_parent_dir(full_path)
                safe_copy_file(fullpath, full_path)
                self.cache_manager.add(full_path)
            except Exception as e:
                logging.exception(f"Failed to cache subtitle file {fullpath}: {e}")
            return
        
        # Handle video file caching
        if need_head:
            length = min(self.head_bytes, metadata['size'])
            if length > 0:
                try:
                    if self.verbose: 
                        logging.info(f"Caching head {length} bytes for {relpath}")
                    ensure_parent_dir(head_path)
                    safe_copy_range(fullpath, head_path, length, offset=0)
                    self.cache_manager.add(head_path)
                except Exception as e:
                    logging.exception(f"Failed to cache head for {fullpath}: {e}")
        
        if need_tail:
            # tail offset
            if metadata['size'] <= self.tail_bytes:
                # small file: tail is entire file (we'll copy entire)
                offset = 0
                length = metadata['size']
            else:
                offset = metadata['size'] - self.tail_bytes
                length = self.tail_bytes
            if length > 0:
                try:
                    if self.verbose: 
                        logging.info(f"Caching tail {length} bytes for {relpath}")
                    ensure_parent_dir(tail_path)
                    safe_copy_range(fullpath, tail_path, length, offset=offset)
                    self.cache_manager.add(tail_path)
                except Exception as e:
                    logging.exception(f"Failed to cache tail for {fullpath}: {e}")


    def _check_cache_database(self):
        """Check database of pending files to cache.
        
        This allows you to add files to be cached without causing HDD spin-up.
        Files are added to the database by external processes or scripts.
        """
        if not os.path.exists(self.cache_db_path):
            # Create empty database if it doesn't exist
            with open(self.cache_db_path, 'w') as f:
                json.dump({"pending": []}, f)
            return
            
        try:
            with open(self.cache_db_path, 'r') as f:
                db = json.load(f)
                
            pending = db.get("pending", [])
            if not pending:
                return
                
            if self.verbose:
                logging.info(f"Found {len(pending)} files in cache database to process")
                
            # Process pending files
            remaining = []
            for entry in pending:
                if isinstance(entry, str):
                    # Simple path format
                    path = entry
                    if os.path.exists(path) and os.path.isfile(path):
                        try:
                            rel = os.path.relpath(path, self.backing_root)
                            self._ensure_cache_for(path, rel)
                            if self.verbose:
                                logging.info(f"Cached file from database: {path}")
                        except Exception as e:
                            logging.warning(f"Failed to cache file from database {path}: {e}")
                            remaining.append(path)
                elif isinstance(entry, dict):
                    # Extended format with attributes
                    path = entry.get("path")
                    if path and os.path.exists(path) and os.path.isfile(path):
                        try:
                            rel = os.path.relpath(path, self.backing_root)
                            self._ensure_cache_for(path, rel)
                            if self.verbose:
                                logging.info(f"Cached file from database: {path}")
                        except Exception as e:
                            logging.warning(f"Failed to cache file from database {path}: {e}")
                            remaining.append(entry)
                    else:
                        remaining.append(entry)
                        
            # Update database with remaining files
            with open(self.cache_db_path, 'w') as f:
                json.dump({"pending": remaining}, f)
                
            if self.verbose and len(remaining) < len(pending):
                logging.info(f"Processed {len(pending) - len(remaining)} files from cache database")
                
        except Exception as e:
            logging.exception(f"Error checking cache database: {e}")


# ---------- SSD Cache Filesystem Implementation ----------
class VidCacheFS(MyLoggingMixIn, Operations):
    def __init__(self, backing_root, cache_manager, head_bytes, tail_bytes, 
                 subtitle_extensions, video_extensions,
                 open_spinup_ttl=5, verbose=False, read_only=False, schedule_interval=3600):
        self.backing_root = backing_root
        self.cache = cache_manager
        self.head_bytes = head_bytes
        self.tail_bytes = tail_bytes
        self.subtitle_extensions = subtitle_extensions
        self.video_extensions = video_extensions
        self.open_spinup_ttl = open_spinup_ttl
        self.verbose = verbose
        self.read_only = read_only
        self.fd = 0
        self.open_files = {}
        # Use fixed 60s like original to avoid long stale caches for Docker/Plex
        self.attr_cache_ttl = 60
        self.dir_cache_ttl = 60
        self.attr_cache = {}
        self.attr_cache_lock = threading.RLock()
        self.dir_cache = {}
        self.dir_cache_lock = threading.RLock()
        # Always enable lazy video open by default (no env var needed)
        self.lazy_video_open = True

    def _backing_path(self, path):
        path = path.lstrip('/')
        return os.path.join(self.backing_root, path)

    def _relative_path(self, path):
        path = path.lstrip('/')
        return path

    def _relpath(self, path):
        if path.startswith('/'):
            return path[1:]
        return path

    def _cache_paths(self, relpath):
        _, ext = os.path.splitext(relpath.lower())
        is_subtitle = ext in self.subtitle_extensions
        
        if is_subtitle:
            full = self.cache.get_path_for(relpath, 'full')
            meta = self.cache.get_path_for(relpath, 'meta')
            # For consistency with the interface, return these as head/tail
            return None, None, meta, full
        else:
            head = self.cache.get_path_for(relpath, 'head')
            tail = self.cache.get_path_for(relpath, 'tail')
            meta = self.cache.get_path_for(relpath, 'meta')
            return head, tail, meta, None

    # New helper for lazy open logic (was missing)
    def _should_lazy_open_video(self, rel, acc_mode):
        if not self.lazy_video_open:
            return False
        if acc_mode != os.O_RDONLY:
            return False
        _, ext = os.path.splitext(rel.lower())
        if ext not in self.video_extensions:
            return False
        head_path, _, _, _ = self._cache_paths(rel)
        return os.path.exists(head_path)

    # Added helper methods for cache invalidation
    def _invalidate_attr_cache(self, path):
        try:
            with self.attr_cache_lock:
                self.attr_cache.pop(path, None)
        except Exception:
            pass

    def _invalidate_dir_cache(self, path):
        try:
            with self.dir_cache_lock:
                self.dir_cache.pop(path, None)
        except Exception:
            pass

    # ---------- Filesystem methods ----------
    def getattr(self, path, fh=None):
        rel = self._relpath(path)
        
        # For ANY path, check our in-memory attribute cache first
        with self.attr_cache_lock:
            if path in self.attr_cache:
                attr_dict, timestamp = self.attr_cache[path]
                # Check if cache entry is still valid
                if time.time() - timestamp < self.attr_cache_ttl:
                    return attr_dict
        
        # Check if it's a directory - but use backing storage only if needed
        backing = self._backing_path(path)
        if os.path.isdir(backing):
            try:
                st = os.lstat(backing)
                attr_dict = dict((key, getattr(st, key)) for key in (
                    'st_atime', 'st_ctime', 'st_gid', 'st_mode', 'st_mtime',
                    'st_nlink', 'st_size', 'st_uid'))
                
                # Cache directory attributes too
                with self.attr_cache_lock:
                    self.attr_cache[path] = (attr_dict, time.time())
                
                return attr_dict
            except FileNotFoundError:
                raise OSError(errno.ENOENT, '')
        with self.attr_cache_lock:
            if path in self.attr_cache:
                attr_dict, timestamp = self.attr_cache[path]
                # Check if cache entry is still valid
                if time.time() - timestamp < self.attr_cache_ttl:
                    return attr_dict
        
        # Next, check if we have metadata cached on disk
        _, ext = os.path.splitext(rel.lower())
        is_subtitle = ext in self.subtitle_extensions
        is_video = ext in self.video_extensions
        
        if is_subtitle or is_video:
            # Get cache paths
            _, _, meta_path, _ = self._cache_paths(rel)
            
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                        
                    # If the metadata has complete attributes, use them
                    if all(key in metadata for key in ('size', 'mtime', 'atime', 'ctime', 'mode', 'uid', 'gid', 'nlink')):
                        attr_dict = {
                            'st_size': metadata['size'],
                            'st_mtime': metadata['mtime'],
                            'st_atime': metadata['atime'],
                            'st_ctime': metadata['ctime'],
                            'st_mode': metadata['mode'],
                            'st_uid': metadata['uid'],
                            'st_gid': metadata['gid'],
                            'st_nlink': metadata['nlink']
                        }
                        
                        # Store in memory cache
                        with self.attr_cache_lock:
                            self.attr_cache[path] = (attr_dict, time.time())
                        
                        return attr_dict
                except Exception:
                    # Fall through to backing access on failure
                    pass
        
        # If we got here, we need to access the backing storage
        try:
            st = os.lstat(backing)
            attr_dict = dict((key, getattr(st, key)) for key in (
                'st_atime', 'st_ctime', 'st_gid', 'st_mode', 'st_mtime',
                'st_nlink', 'st_size', 'st_uid'))
            
            # Store in memory cache
            with self.attr_cache_lock:
                self.attr_cache[path] = (attr_dict, time.time())
            
            # Update the metadata file if this is a file type we cache
            if is_subtitle or is_video:
                _, _, meta_path, _ = self._cache_paths(rel)
                try:
                    ensure_parent_dir(meta_path)
                    with open(meta_path, 'w') as f:
                        json.dump({
                            'size': st.st_size,
                            'mtime': st.st_mtime,
                            'atime': st.st_atime,
                            'ctime': st.st_ctime,
                            'mode': st.st_mode,
                            'uid': st.st_uid,
                            'gid': st.st_gid,
                            'nlink': st.st_nlink
                        }, f)
                except Exception as e:
                    if self.verbose:
                        logging.warning(f"Failed to update metadata cache for {path}: {e}")
            
            return attr_dict
        except FileNotFoundError:
            raise OSError(errno.ENOENT, '')

    def readdir(self, path, fh):
        # Check cache first
        with self.dir_cache_lock:
            if path in self.dir_cache:
                dirents, timestamp = self.dir_cache[path]
                # Check if cache entry is still valid
                if time.time() - timestamp < self.dir_cache_ttl:
                    if self.verbose:
                        logging.debug(f"Serving directory listing from cache: {path}")
                    for r in dirents:
                        yield r
                    return

        # If not in cache or expired, access backing storage
        backing = self._backing_path(path)
        dirents = ['.', '..']
        if os.path.isdir(backing):
            dirents.extend(os.listdir(backing))
        
        # Save to cache
        with self.dir_cache_lock:
            self.dir_cache[path] = (dirents, time.time())
        
        # Return listing
        for r in dirents:
            yield r

    def readlink(self, path):
        pathname = os.readlink(self._backing_path(path))
        return pathname

    def access(self, path, mode):
        # Serve from cached metadata first to avoid spinning up disks
        rel = self._relpath(path)
        _, ext = os.path.splitext(rel.lower())
        if ext in self.subtitle_extensions or ext in self.video_extensions:
            _, _, meta_path, _ = self._cache_paths(rel)
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    perm_mode = meta.get('mode')
                    owner = meta.get('uid')
                    group = meta.get('gid')
                    if perm_mode is not None and owner is not None and group is not None:
                        if (mode & os.W_OK) and self.read_only:
                            raise OSError(errno.EROFS, 'Read-only file system')
                        euid = os.geteuid() if hasattr(os, 'geteuid') else -1
                        egroups = set(os.getgroups()) if hasattr(os, 'getgroups') else set()
                        # Determine which permission set applies
                        if owner == euid:
                            mask_r = stat.S_IRUSR; mask_w = stat.S_IWUSR; mask_x = stat.S_IXUSR
                        elif group in egroups:
                            mask_r = stat.S_IRGRP; mask_w = stat.S_IWGRP; mask_x = stat.S_IXGRP
                        else:
                            mask_r = stat.S_IROTH; mask_w = stat.S_IWOTH; mask_x = stat.S_IXOTH
                        def check(bit, m):
                            return (bit & mode) == 0 or (perm_mode & m) != 0
                        if (check(os.R_OK, mask_r) and
                            check(os.W_OK, mask_w) and
                            check(os.X_OK, mask_x)):
                            return  # access granted via metadata
                        raise OSError(errno.EACCES, '')
                except OSError:
                    raise
                except Exception:
                    pass  # Fallback to backing check if metadata malformed
        # Fallback: direct backing access
        backing = self._backing_path(path)
        if not os.access(backing, mode):
            raise OSError(errno.EACCES, '')

    def chmod(self, path, mode):
        if self.read_only:
            raise OSError(errno.EROFS, 'Read-only file system')
        backing = self._backing_path(path)
        result = os.chmod(backing, mode)
        self._invalidate_attr_cache(path)
        return result

    def chown(self, path, uid, gid):
        if self.read_only:
            raise OSError(errno.EROFS, 'Read-only file system')
        backing = self._backing_path(path)
        result = os.chown(backing, uid, gid)
        self._invalidate_attr_cache(path)
        return result

    def statfs(self, path):
        backing = self._backing_path(path)
        stv = os.statvfs(backing)
        return dict((key, getattr(stv, key)) for key in (
            'f_bavail', 'f_bfree', 'f_blocks', 'f_bsize', 'f_favail',
            'f_ffree', 'f_files', 'f_flag', 'f_frsize', 'f_namemax'))

    def open(self, path, flags):
        rel = self._relpath(path)
        _, ext = os.path.splitext(rel.lower())
        is_subtitle = ext in self.subtitle_extensions
        backing = self._backing_path(path)
        acc_mode = os.O_RDONLY | (flags & os.O_WRONLY) | (flags & os.O_RDWR)
        if acc_mode != os.O_RDONLY and self.read_only:
            raise OSError(errno.EROFS, 'Read-only file system')
        self.fd += 1
        # Subtitle lazy open (serve from full cache if present)
        if is_subtitle and acc_mode == os.O_RDONLY:
            _, _, _, full_path = self._cache_paths(rel)
            if os.path.exists(full_path):
                self.cache.touch(full_path)
                self.open_files[self.fd] = (None, path, 'subtitle-lazy')
                if self.open_spinup_ttl > 0:
                    threading.Thread(target=self._ensure_spinup, args=(backing, self.open_spinup_ttl), daemon=True).start()
                return self.fd
        # Video lazy open if head cached
        if self._should_lazy_open_video(rel, acc_mode):
            self.open_files[self.fd] = (None, path, 'video-lazy')
            if self.open_spinup_ttl > 0:
                threading.Thread(target=self._ensure_spinup, args=(backing, self.open_spinup_ttl), daemon=True).start()
            return self.fd
        # Fallback: open backing immediately
        fd = os.open(backing, flags)
        self.open_files[self.fd] = (fd, path, 'direct')
        if self.open_spinup_ttl > 0:
            threading.Thread(target=self._ensure_spinup, args=(backing, self.open_spinup_ttl), daemon=True).start()
        return self.fd

    def _ensure_spinup(self, backing_path, ttl):
        """Ensure backing HDDs are spinning up by actively reading data from the file."""
        try:
            if self.verbose:
                logging.debug(f"Background spinup for {backing_path}")
            
            # Open the file WITHOUT O_NONBLOCK to ensure disk access is initiated
            fd = os.open(backing_path, os.O_RDONLY)
            
            try:
                # Get file size
                file_size = os.fstat(fd).st_size
                
                # Read a small chunk from the start of the file
                os.lseek(fd, 0, os.SEEK_SET)
                _ = os.read(fd, 64 * 1024)  # Read 64KB
                
                # If it's a large file, also read from the middle to ensure disk is fully spinning
                if file_size > 1 * 1024 * 1024:  # If larger than 1MB
                    middle_pos = file_size // 2
                    os.lseek(fd, middle_pos, os.SEEK_SET)
                    _ = os.read(fd, 64 * 1024)  # Read another 64KB
                
                # Keep the file open for the specified TTL to ensure drive stays spinning
                if ttl > 0:
                    time.sleep(ttl)
                    
                if self.verbose:
                    logging.debug(f"Spinup complete for {backing_path}")
            finally:
                os.close(fd)
        except Exception as e:
            # Log the error but don't fail the open operation
            if self.verbose:
                logging.debug(f"Spinup error for {backing_path}: {str(e)}")

    def read(self, path, size, offset, fh):
        rel = self._relpath(path)
        _, ext = os.path.splitext(rel.lower())
        is_subtitle = ext in self.subtitle_extensions
        
        # Add detailed debug logging for all read operations
        if self.verbose:
            logging.debug(f"READ request: path={path}, size={size}, offset={offset}, fh={fh}")
        
        # Get cache paths
        head_path, tail_path, meta_path, full_path = self._cache_paths(rel)
        
        if self.verbose:
            logging.debug(f"Cache paths: head={head_path}, tail={tail_path}, meta={meta_path}, full={full_path}")
            logging.debug(f"Cache exists: head={os.path.exists(head_path)}, tail={os.path.exists(tail_path)}, meta={os.path.exists(meta_path)}")
        
        # First try to get file size from cached metadata
        fsize = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                    fsize = metadata.get('size')
                    if self.verbose:
                        logging.debug(f"Got file size from metadata: {fsize}")
            except Exception as e:
                if self.verbose:
                    logging.debug(f"Failed to read metadata: {e}")
        
        # Handle subtitle files differently - we cache them fully
        if is_subtitle:
            if full_path and os.path.exists(full_path):
                if self.verbose:
                    logging.debug(f"Serving subtitle from full cache: {path}")
                with open(full_path, 'rb') as f:
                    f.seek(offset)
                    data = f.read(size)
                    self.cache.touch(full_path)
                    return data
    
        # Get information about this file handle
        fd, _, fd_type = self.open_files[fh]
    
        # For video files, check if data is in the cache
        if not is_subtitle:
            # If we have file size from metadata, use it
            if fsize is not None:
                # Check if request is entirely within head cache
                if offset < self.head_bytes and os.path.exists(head_path):
                    head_avail = min(self.head_bytes, fsize) - offset
                    if head_avail > 0:
                        if self.verbose:
                            logging.debug(f"Attempting to read {min(size, head_avail)} bytes from head cache at offset {offset}")
                        with open(head_path, 'rb') as h:
                            h.seek(offset)
                            data = h.read(min(size, head_avail))
                            self.cache.touch(head_path)
                            if len(data) == size or offset + len(data) >= fsize:
                                if self.verbose:
                                    logging.debug(f"Successfully served {len(data)} bytes from head cache")
                                return data
                            # Need more data from backing file - open if needed
                            if fd is None:
                                backing = self._backing_path(path)
                                if self.verbose:
                                    logging.debug(f"Opening backing file for additional data: {backing}")
                                fd = os.open(backing, os.O_RDONLY)
                                self.open_files[fh] = (fd, path, fd_type)
                            # Read remaining from backing
                            remaining = size - len(data)
                            if self.verbose:
                                logging.debug(f"Reading remaining {remaining} bytes from backing file")
                            os.lseek(fd, offset + len(data), os.SEEK_SET)
                            more = os.read(fd, remaining)
                            return data + more

                # Calculate tail cache parameters
                if fsize <= self.tail_bytes:
                    tail_start = 0
                else:
                    tail_start = fsize - self.tail_bytes
            
                # Debug logging for tail cache checks
                if self.verbose:
                    bytes_from_end = fsize - offset
                    logging.debug(f"Tail params: fsize={fsize}, tail_bytes={self.tail_bytes}, tail_start={tail_start}")
                    logging.debug(f"Offset check: offset={offset}, bytes_from_end={bytes_from_end}")
                    logging.debug(f"Should use tail cache: {offset >= tail_start}")
                    if os.path.exists(tail_path):
                        tail_size = os.path.getsize(tail_path)
                        logging.debug(f"Tail cache file size: {tail_size}")
                        
                        # Check if this request would be in bounds of the tail file
                        if offset >= tail_start:
                            tail_offset = offset - tail_start
                            logging.debug(f"Calculated tail_offset={tail_offset}, valid range: 0-{tail_size-1}")
                            if 0 <= tail_offset < tail_size:
                                logging.debug("Tail offset is in valid range")
                            else:
                                logging.debug("Tail offset is OUT OF RANGE!")
            
            # Check if request is entirely within tail cache
            if offset >= tail_start and os.path.exists(tail_path):
                # Get actual size of tail cache file for better bounds checking
                try:
                    actual_tail_size = os.path.getsize(tail_path)
                    
                    # Calculate precise tail offset
                    tail_offset = offset - tail_start
                    
                    if self.verbose:
                        logging.debug(f"Attempting to read from tail cache: offset={offset}, tail_start={tail_start}, tail_offset={tail_offset}")
                        logging.debug(f"Actual tail cache size: {actual_tail_size}")
                    
                    # Ensure tail_offset is within bounds
                    if 0 <= tail_offset < actual_tail_size:
                        with open(tail_path, 'rb') as t:
                            t.seek(tail_offset)
                            data = t.read(size)
                            self.cache.touch(tail_path)
                            if self.verbose:
                                logging.debug(f"Successfully read {len(data)} bytes from tail cache")
                            return data
                    else:
                        if self.verbose:
                            logging.debug(f"Tail offset {tail_offset} out of bounds (0-{actual_tail_size-1})")
                except Exception as e:
                    if self.verbose:
                        logging.debug(f"Error reading from tail cache: {e}")
    
        # If we get here, we need to access the backing file
        # Open it now if it's not already open
        if fd is None:
            backing = self._backing_path(path)
            if self.verbose:
                logging.debug(f"Cache miss - opening backing file: {backing}")
            fd = os.open(backing, os.O_RDONLY)
            self.open_files[fh] = (fd, path, fd_type)
        
        # Fallback to backing file
        if self.verbose:
            logging.debug(f"Reading {size} bytes from backing file at offset {offset}")
        os.lseek(fd, offset, os.SEEK_SET)
        data = os.read(fd, size)
        if self.verbose:
            logging.debug(f"Read {len(data)} bytes from backing file")
        return data    
    def write(self, path, data, offset, fh):
        if self.read_only:
            raise OSError(errno.EROFS, 'Read-only file system')
        fd, _, _ = self.open_files[fh]
        if fd is None:
            # Need a real fd now for write
            backing = self._backing_path(path)
            fd = os.open(backing, os.O_RDWR)
            self.open_files[fh] = (fd, path, 'promoted')
        os.lseek(fd, offset, os.SEEK_SET)
        result = os.write(fd, data)
        self._invalidate_attr_cache(path)
        return result

    def flush(self, path, fh):
        fd, _, _ = self.open_files[fh]
        if fd is None:
            return 0
        return os.fsync(fd)

    def release(self, path, fh):
        fd, _, _ = self.open_files.pop(fh)
        if fd is not None:
            return os.close(fd)
        return 0

    def fsync(self, path, datasync, fh):
        fd, _, _ = self.open_files[fh]
        if fd is None:
            return 0
        return os.fsync(fd)

    def truncate(self, path, length, fh=None):
        if self.read_only:
            raise OSError(errno.EROFS, 'Read-only file system')
        backing = self._backing_path(path)
        with open(backing, 'r+') as f:
            f.truncate(length)
            
        # Invalidate attribute cache
        self._invalidate_attr_cache(path)
            
        # Delete cached files since they're now invalid
        rel = self._relative_path(path)
        head_path, tail_path, meta_path, full_path = self._cache_paths(rel)
        
        for cache_file in [head_path, tail_path, meta_path, full_path]:
            if cache_file and os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                    if self.verbose:
                        logging.info(f"Removed cache file after truncate: {cache_file}")
                except OSError as e:
                    logging.warning(f"Failed to remove cache file {cache_file}: {e}")

    def utimens(self, path, times=None):
        backing = self._backing_path(path)
        result = os.utime(backing, times)
        self._invalidate_attr_cache(path)
        return result

    def create(self, path, mode, fi=None):
        if self.read_only:
            raise OSError(errno.EROFS, 'Read-only file system')
        backing = self._backing_path(path)
        fd = os.open(backing, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        self.fd += 1
        self.open_files[self.fd] = (fd, path, 'create')
        self._invalidate_dir_cache(os.path.dirname(path))
        return self.fd

    def mkdir(self, path, mode):
        if self.read_only:
            raise OSError(errno.EROFS, 'Read-only file system')
        backing = self._backing_path(path)
        os.mkdir(backing, mode)
        self._invalidate_dir_cache(os.path.dirname(path))
        
    def unlink(self, path):
        if self.read_only:
            raise OSError(errno.EROFS, 'Read-only file system')
        backing = self._backing_path(path)
        # Delete the backing file
        os.unlink(backing)
        
        # Invalidate caches
        self._invalidate_attr_cache(path)
        self._invalidate_dir_cache(os.path.dirname(path))
        
        # Clean up associated cache files
        rel = self._relative_path(path)
        head_path, tail_path, meta_path, full_path = self._cache_paths(rel)
        
        # Remove cached files if they exist
        for cache_file in [head_path, tail_path, meta_path, full_path]:
            if cache_file and os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                    if self.verbose:
                        logging.info(f"Deleted cached file for {path}: {cache_file}")
                except OSError as e:
                    logging.warning(f"Failed to delete cached file {cache_file}: {e}")
                    
    def rmdir(self, path):
        if self.read_only:
            raise OSError(errno.EROFS, 'Read-only file system')
        backing = self._backing_path(path)
        os.rmdir(backing)
        
        # Invalidate directory caches
        self._invalidate_dir_cache(path)
        self._invalidate_dir_cache(os.path.dirname(path))
        
        # Note: We don't need to clean up cache files here since directories 
        # themselves don't have cache entries, only files within them do.
        # When a file is deleted, its cache files are cleaned up by unlink().
        
    def rename(self, old, new):
        if self.read_only:
            raise OSError(errno.EROFS, 'Read-only file system')
        old_backing = self._backing_path(old)
        new_backing = self._backing_path(new)
        
        # Rename the backing file
        os.rename(old_backing, new_backing)
        
        # Invalidate caches
        self._invalidate_attr_cache(old)
        self._invalidate_attr_cache(new)
        self._invalidate_dir_cache(os.path.dirname(old))
        self._invalidate_dir_cache(os.path.dirname(new))
        
        # Handle cached files - for simplicity, we'll delete the old cache files
        # The new file will be cached on access or by the scheduler
        old_rel = self._relative_path(old)
        old_head, old_tail, old_meta, old_full = self._cache_paths(old_rel)
        
        # Clean up old cache files
        for cache_file in [old_head, old_tail, old_meta, old_full]:
            if cache_file and os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                    if self.verbose:
                        logging.info(f"Deleted old cache file during rename: {cache_file}")
                except OSError as e:
                    logging.warning(f"Failed to delete cache file during rename {cache_file}: {e}")


# ---------- Process manager for multiple mounts ----------
class MultiMountManager:
    def __init__(self, config, foreground=False, global_read_only=False):
        """Initialize the mount manager.
        
        Args:
            config: Configuration dictionary or path to config file
            foreground: If True, run in foreground mode
            global_read_only: If True, all mounts will be read-only
        """
        self.foreground = foreground
        self.global_read_only = global_read_only
        self.processes = []
        self.total_estimated_cache = 0  # Keep track of total estimated cache
        
        # Handle both direct config dict and config path
        if isinstance(config, dict):
            self.config = config
        else:
            self.config_path = config
            self.load_config()
    
    def _check_overprovisioning(self, mount_config, mount_cache_size, global_max_cache):
        """Check if this mount contributes to overprovisioning and log a warning if needed."""
        # Add this mount's estimated cache to the total
        self.total_estimated_cache += mount_cache_size
        
        # Calculate the percentage of the global cache that is used
        usage_percent = (self.total_estimated_cache / global_max_cache) * 100
        
        # Log information about this mount's usage
        backing = mount_config['BACKING_DIR']
        logging.info(f"Mount {backing} adds {mount_cache_size/(1024*1024*1024):.2f} GB to total cache usage")
        logging.info(f"Total estimated cache usage now: {self.total_estimated_cache/(1024*1024*1024):.2f} GB of {global_max_cache/(1024*1024*1024):.2f} GB ({usage_percent:.1f}%)")
        
        # Warn if we're overprovisioning
        if self.total_estimated_cache > global_max_cache:
            over_percent = ((self.total_estimated_cache - global_max_cache) / global_max_cache) * 100
            logging.warning(f"WARNING: Total estimated cache usage exceeds global maximum by {over_percent:.1f}%")
            logging.warning(f"Cache overprovisioned by {(self.total_estimated_cache - global_max_cache)/(1024*1024*1024):.2f} GB")
            logging.warning("This will cause more LRU evictions but is allowed")
    
    def load_config(self):
        """Load configuration from the config path."""
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Process numeric expressions in global settings
        if 'GLOBAL' in self.config and 'MAX_CACHE_SIZE_BYTES' in self.config['GLOBAL']:
            if isinstance(self.config['GLOBAL']['MAX_CACHE_SIZE_BYTES'], str):
                self.config['GLOBAL']['MAX_CACHE_SIZE_BYTES'] = eval(self.config['GLOBAL']['MAX_CACHE_SIZE_BYTES'])
        
        # Process numeric expressions in each mount config
        for mount in self.config['MOUNTS']:
            for key in ['HEAD_BYTES', 'TAIL_BYTES', 'MAX_CACHE_SIZE_BYTES']:
                if key in mount and isinstance(mount[key], str):
                    mount[key] = eval(mount[key])
    def _mount_filesystem(self, mount_config):
        pid = os.getpid()
        backing = mount_config['BACKING_DIR']
        logging.info(f"Process {pid}: Starting filesystem mount for {backing}")
        
        # Get mount-specific settings, or fallback to global settings
        mountpoint = mount_config['MOUNT_POINT']
        cache_dir = mount_config['CACHE_DIR']
        
        # Ensure numeric values for cache sizes
        head_bytes = mount_config.get('HEAD_BYTES', 60 * 1024 * 1024)  
        if isinstance(head_bytes, str):
            head_bytes = eval(head_bytes)
        
        tail_bytes = mount_config.get('TAIL_BYTES', 1 * 1024 * 1024)
        if isinstance(tail_bytes, str):
            tail_bytes = eval(tail_bytes)
    
        # Calculate the max cache size for this mount based on max files and head/tail bytes
        max_files = mount_config.get('MAX_FILES', 100)  # Default to 100 files if not specified
        
        # Calculate estimated cache size based on max files
        # Formula: max_files * (head_bytes + tail_bytes + metadata overhead)
        metadata_overhead = 1024  # Approximately 1KB for metadata per file
        estimated_bytes_per_file = head_bytes + tail_bytes + metadata_overhead
        max_cache = max_files * estimated_bytes_per_file
        
        # Get global max cache for warning about overprovisioning
        global_max_cache = self.config['GLOBAL'].get('MAX_CACHE_SIZE_BYTES', 100 * 1024 * 1024 * 1024)
        
        # Log the estimated cache size and files count
        logging.info(f"Mount {backing} configured for up to {max_files} files")
        logging.info(f"Estimated cache usage: {max_cache/(1024*1024*1024):.2f} GB ({max_cache/(1024*1024)} MB)")
        
        # Calculate if this mount contributes to overprovisioning
        self._check_overprovisioning(mount_config, max_cache, global_max_cache)
        
        schedules = mount_config.get('SCHEDULES', [])
        open_ttl = mount_config.get('OPEN_SPINUP_TTL', 5)
        verbose = mount_config.get('VERBOSE', self.config['GLOBAL'].get('VERBOSE', False))
        
        # Get subtitle and video extensions
        subtitle_extensions = self.config.get('SUBTITLE_EXTENSIONS', 
                                             [".srt", ".vtt", ".sub", ".ass", ".ssa", ".idx", ".smi"])
        video_extensions = self.config.get('VIDEO_EXTENSIONS', 
                                          [".mkv", ".mp4", ".avi", ".mov", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".webm"])
        
        # Check if file watching is enabled for this mount
        enable_file_watching = mount_config.get('ENABLE_FILE_WATCHING', 
                                                self.config['GLOBAL'].get('ENABLE_FILE_WATCHING', True))
        
        # Check if read-only mode is specified for this mount or globally
        read_only = mount_config.get('READ_ONLY', self.config['GLOBAL'].get('READ_ONLY', False)) or self.global_read_only
          # Validate directories
        if not os.path.isdir(backing):
            logging.error(f"Backing dir {backing} does not exist")
            return None
            
        # Create cache directory if it doesn't exist
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        # Prepare mount point using the robust preparation function
        if not self._prepare_mount_point(mountpoint):
            logging.error(f"Could not prepare mount point {mountpoint}, aborting mount")
            return None
            
        # Add a small delay to ensure the mount point is fully released
        time.sleep(0.5)
    
        # Initialize cache manager
        cache_mgr = CacheManager(cache_dir, max_cache, verbose=verbose)

        # One-time reconciliation (size mismatch + max_files prune)
        try:
            cache_mgr.reconcile_startup(head_bytes, tail_bytes, max_files)
        except Exception as e:
            if verbose:
                logging.warning(f"Startup reconciliation failed: {e}")
        
        # Initialize scheduler for background caching
        use_cache_db = mount_config.get('USE_CACHE_DB', self.config['GLOBAL'].get('USE_CACHE_DB', False))
        check_file_modifications = mount_config.get('CHECK_FILE_MODIFICATIONS', 
                                                   self.config['GLOBAL'].get('CHECK_FILE_MODIFICATIONS', False))
        
        scheduler = Scheduler(schedules, cache_mgr, head_bytes, tail_bytes, backing, 
                            subtitle_extensions, video_extensions, verbose=verbose,
                            use_cache_db=use_cache_db, check_file_modifications=check_file_modifications)
        scheduler.start()
        
        # Initialize file watcher for real-time caching of new files
        file_watcher = None
        if enable_file_watching and INOTIFY_AVAILABLE:
            file_watcher = FileWatcher(backing, cache_mgr, head_bytes, tail_bytes,
                                    subtitle_extensions, video_extensions, verbose=verbose)
            file_watcher.start()
            if verbose:
                logging.info(f"Started file watcher for {backing}")
        
        # Get schedule interval (use the shortest interval if multiple schedules)
        schedule_interval = 3600  # Default 1 hour
        if schedules:
            intervals = [s.get('scan_interval_seconds', 3600) for s in schedules]
            schedule_interval = min(intervals)  # Use the shortest interval
    
        # Initialize FUSE filesystem with schedule interval
        fuse_ops = VidCacheFS(backing, cache_mgr, head_bytes, tail_bytes, 
                            subtitle_extensions, video_extensions,
                            open_spinup_ttl=open_ttl, verbose=verbose, 
                            read_only=read_only, schedule_interval=schedule_interval)
        
        # FUSE mounting options
        fuse_options = {}
        fuse_options['allow_other'] = True
        if read_only:
            fuse_options['ro'] = True
            logging.info(f"Mounting {backing} at {mountpoint} in read-only mode")
        else:
            logging.info(f"Mounting {backing} at {mountpoint}")
        
        # Always run in foreground mode within the child process to prevent daemonization
        # This is critical for the child process to work properly
        foreground_mode = True
        
        try:
            logging.info(f"Process {pid}: About to start FUSE mount for {backing}")
            FUSE(fuse_ops, mountpoint, nothreads=True, foreground=foreground_mode, **fuse_options)
            logging.info(f"Process {pid}: FUSE mount completed for {backing}")
        except Exception as e:
            logging.exception(f"Process {pid}: FUSE mount failed for {backing}")
            scheduler.stop()
            if file_watcher:
                file_watcher.stop()
            return None
            
        return (scheduler, file_watcher)
    
    def _process_mount_wrapper(self, mount_config):
        """Wrapper function for processing mount in a separate process."""
        try:
            # Validate configuration again in the child process for safety
            if not self._validate_mount_config(mount_config):
                logging.error(f"Invalid mount configuration detected in child process, aborting mount for {mount_config['BACKING_DIR']}")
                return None
            
            return self._mount_filesystem(mount_config)
        except Exception as e:
            logging.exception(f"Error in mount process: {e}")
            return None
    
    def mount_all(self):
        """Mount all filesystems defined in the config."""
        if self.foreground:
            # In foreground mode, just mount the first filesystem
            if len(self.config['MOUNTS']) > 0:
                mount_config = self.config['MOUNTS'][0]
                # Validate the configuration
                if not self._validate_mount_config(mount_config):
                    logging.error(f"Invalid mount configuration, aborting mount for {mount_config['BACKING_DIR']}")
                    return
                
                logging.info(f"Mounting in foreground mode: {mount_config['BACKING_DIR']}")
                scheduler = self._mount_filesystem(mount_config)
                if scheduler:
                    self.processes.append(scheduler)
        else:
            # In background mode, launch each mount in its own process
            for mount_config in self.config['MOUNTS']:
                # Validate the configuration before starting the process
                if not self._validate_mount_config(mount_config):
                    logging.error(f"Invalid mount configuration, skipping mount for {mount_config['BACKING_DIR']}")
                    continue
                
                logging.info(f"Starting mount process for {mount_config['BACKING_DIR']}")
                p = multiprocessing.Process(
                    target=self._process_mount_wrapper,
                    args=(mount_config,)
                )
                p.daemon = False
                p.start()
                self.processes.append(p)
                logging.info(f"Process started with PID {p.pid}")
        
            # In background mode, keep the main process alive
            try:
                while True:
                    alive_processes = [p for p in self.processes 
                                      if isinstance(p, multiprocessing.Process) and p.is_alive()]
                
                    if not alive_processes:
                        logging.warning("All mount processes have terminated")
                        break
                    
                    time.sleep(1)
            except KeyboardInterrupt:
                logging.info("Main process received keyboard interrupt")
    
    def stop_all(self):
        """Stop all mounted filesystems and ensure proper cleanup."""
        process_name = multiprocessing.current_process().name
        
        logging.info(f"Stop all called from process: {process_name}")
        
        # Only the main process should try to terminate child processes
        if process_name == 'MainProcess':
            for p in self.processes:
                if isinstance(p, threading.Thread):
                    try:
                        p.stop()
                    except:
                        pass
                elif isinstance(p, multiprocessing.Process):
                    try:
                        p.terminate()
                        p.join(timeout=2)
                    except AssertionError:
                        # Skip if we can't join the process
                        logging.warning(f"Could not join process - may not be a child of current process")
                    except Exception as e:
                        logging.warning(f"Error stopping process: {e}")
                        
            # Clean up any stale mounts
            self._unmount_stale_points()
        else:
            # In child process, just clean up the threads
            for p in self.processes:
                if isinstance(p, threading.Thread):
                    try:
                        p.stop()
                    except:
                        pass
    
        logging.info(f"All mounts stopped in process {process_name}")

    def _unmount_stale_points(self):
        """Unmount any stale mount points to ensure clean restart."""
        for mount_config in self.config['MOUNTS']:
            mountpoint = mount_config['MOUNT_POINT']
            if os.path.exists(mountpoint) and os.path.ismount(mountpoint):
                self._unmount_point(mountpoint)

    def _unmount_point(self, mountpoint):
        """Unmount a specific mount point.
        
        Returns:
            bool: True if successful or if already unmounted, False otherwise
        """
        if not os.path.ismount(mountpoint):
            return True  # Already unmounted
            
        # Try up to 3 times to unmount
        for attempt in range(3):
            try:
                logging.info(f"Unmounting {mountpoint} (attempt {attempt+1})")
                
                # Try fusermount first (preferred method)
                try:
                    subprocess.run(['fusermount', '-u', '-z', mountpoint], 
                                  check=False, timeout=5,
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
                except (subprocess.SubprocessError, FileNotFoundError):
                    try:
                        # Fallback to umount with force and lazy options
                        subprocess.run(['umount', '-f', '-l', mountpoint], 
                                      check=False, timeout=5,
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE)
                    except Exception as e:
                        logging.warning(f"Both unmount methods failed for {mountpoint}: {e}")
                
                # Check if unmount was successful
                if not os.path.ismount(mountpoint):
                    logging.info(f"Successfully unmounted {mountpoint}")
                    # Wait a moment for the system to fully release the mount
                    time.sleep(0.5)
                    return True
                    
            except Exception as e:
                logging.warning(f"Error during unmount attempt {attempt+1} for {mountpoint}: {e}")
            
            # Wait between attempts
            time.sleep(1)
        
        # If we got here, all unmount attempts failed
        logging.error(f"Failed to unmount {mountpoint} after multiple attempts")
        return False

    def _prepare_mount_point(self, mountpoint):
        """Prepare a mount point by ensuring it's unmounted and exists as a directory.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # First, check if the mount point is already a mount
        if os.path.ismount(mountpoint):
            logging.info(f"Mount point {mountpoint} is already mounted, attempting to unmount")
            # Try to unmount it
            if not self._unmount_point(mountpoint):
                # If unmount failed, we can't continue
                return False
        
        # Check if it exists but is not a directory
        if os.path.exists(mountpoint) and not os.path.isdir(mountpoint):
            try:
                logging.warning(f"Mount point {mountpoint} exists but is not a directory, removing it")
                os.remove(mountpoint)
            except Exception as e:
                logging.error(f"Failed to remove non-directory mount point {mountpoint}: {e}")
                return False
        
        # Now create the directory if it doesn't exist
        if not os.path.exists(mountpoint):
            try:
                os.makedirs(mountpoint, exist_ok=True)
            except Exception as e:
                logging.error(f"Failed to create mount point directory {mountpoint}: {e}")
                return False
        
        # Verify it's not mounted (double-check)
        if os.path.ismount(mountpoint):
            logging.error(f"Mount point {mountpoint} is still mounted after unmount attempt")
            return False
            
        # Verify we have write access to the mount point
        if not os.access(mountpoint, os.W_OK):
            try:
                # Try to make it writable
                current_mode = os.stat(mountpoint).st_mode
                os.chmod(mountpoint, current_mode | 0o200)  # Add write permission
            except Exception as e:
                logging.warning(f"Could not set write permission on {mountpoint}: {e}")
                # Continue anyway - might still work depending on the system
        
        return True

    def _validate_mount_config(self, mount_config):
        """Validate mount configuration to prevent problematic setups.
    
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        backing_dir = mount_config['BACKING_DIR']
        mount_point = mount_config['MOUNT_POINT']
    
        # 1. Prevent mounting within the backing directory
        if os.path.commonpath([os.path.abspath(backing_dir)]) == os.path.commonpath([os.path.abspath(backing_dir), os.path.abspath(mount_point)]):
            logging.error(f"INVALID CONFIGURATION: Mount point '{mount_point}' is inside backing directory '{backing_dir}'")
            logging.error("This would cause a recursive mount and potentially lock up your system.")
            return False
    
        # 2. Prevent mounting within the cache directory
        cache_dir = mount_config['CACHE_DIR']
        if os.path.commonpath([os.path.abspath(cache_dir)]) == os.path.commonpath([os.path.abspath(cache_dir), os.path.abspath(mount_point)]):
            logging.error(f"INVALID CONFIGURATION: Mount point '{mount_point}' is inside cache directory '{cache_dir}'")
            logging.error("This would cause issues with caching.")
            return False
    
        # 3. Check if the cache directory is inside the backing directory
        if os.path.commonpath([os.path.abspath(backing_dir)]) == os.path.commonpath([os.path.abspath(backing_dir), os.path.abspath(cache_dir)]):
            logging.warning(f"WARNING: Cache directory '{cache_dir}' is inside backing directory '{backing_dir}'")
            logging.warning("This may cause performance issues or unexpected behavior.")
    
        return True
def load_config(config_path=None, use_builtin=False):
    """Load configuration from file or use built-in default.
    
    Args:
        config_path: Path to config file, or None to search in default locations
        use_builtin: If True, use the built-in config regardless of file existence
    
    Returns:
        Loaded configuration dictionary
    """
    # If explicitly using built-in config, return it directly
    if use_builtin:
        logging.info("Using built-in configuration")
        return DEFAULT_CONFIG
    
    # Try the specified config path first if provided
    if config_path:
        try:
            # If path is not absolute, check relative to script directory first
            if not os.path.isabs(config_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                script_relative_path = os.path.join(script_dir, config_path)
                
                if os.path.exists(script_relative_path):
                    config_path = script_relative_path
            
            # Now try to load the config
            with open(config_path, 'r') as f:
                logging.info(f"Loading configuration from {config_path}")
                config = json.load(f)
                return _process_config(config)
        except FileNotFoundError:
            logging.warning(f"Config file not found: {config_path}")
        except json.JSONDecodeError:
            logging.warning(f"Invalid JSON in config file: {config_path}")
        except Exception as e:
            logging.warning(f"Error loading config: {e}")
    
    # If no config path provided or it failed, try standard locations
    for path in CONFIG_PATHS:
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    logging.info(f"Loading configuration from {path}")
                    config = json.load(f)
                    return _process_config(config)
        except Exception:
            continue
    
    # If all else fails, use the built-in default config
    logging.info("No valid configuration file found, using built-in default")
    return DEFAULT_CONFIG

def _process_config(config):
    """Process a loaded config, handle numeric expressions."""
    # Process numeric expressions in global settings
    if 'GLOBAL' in config and 'MAX_CACHE_SIZE_BYTES' in config['GLOBAL']:
        if isinstance(config['GLOBAL']['MAX_CACHE_SIZE_BYTES'], str):
            config['GLOBAL']['MAX_CACHE_SIZE_BYTES'] = eval(config['GLOBAL']['MAX_CACHE_SIZE_BYTES'])
    
    # Ensure CHECK_FILE_MODIFICATIONS is set (default to False)
    if 'GLOBAL' in config:
        config['GLOBAL'].setdefault('CHECK_FILE_MODIFICATIONS', False)
    
    # Process numeric expressions in each mount config
    for mount in config['MOUNTS']:
        for key in ['HEAD_BYTES', 'TAIL_BYTES', 'MAX_CACHE_SIZE_BYTES']:
            if key in mount and isinstance(mount[key], str):
                mount[key] = eval(mount[key])
        
        # Allow per-mount override of CHECK_FILE_MODIFICATIONS
        if 'CHECK_FILE_MODIFICATIONS' not in mount and 'GLOBAL' in config:
            mount.setdefault('CHECK_FILE_MODIFICATIONS', 
                            config['GLOBAL'].get('CHECK_FILE_MODIFICATIONS', False))
    
    return config


# ---------- File Watcher: monitors backing dir for new files ----------
class FileWatcher(threading.Thread):
    def __init__(self, backing_root, cache_manager, head_bytes, tail_bytes, 
                 subtitle_extensions, video_extensions, 
                 verbose=False):
        super().__init__(daemon=True)
        self.backing_root = backing_root
        self.cache_manager = cache_manager
        self.head_bytes = head_bytes
        self.tail_bytes = tail_bytes
        self.subtitle_extensions = subtitle_extensions
        self.video_extensions = video_extensions
        self.verbose = verbose
        self._stop = threading.Event()
        self.watcher = None  # Use a single watcher for the whole tree
        self.max_watches = self._get_max_watches()
        
    def _get_max_watches(self):
        """Get the maximum number of inotify watches allowed by the system."""
        try:
            with open('/proc/sys/fs/inotify/max_user_watches', 'r') as f:
                return int(f.read().strip())
        except Exception:
            # Default to a reasonable value if we can't read the system limit
            return 8192
        
    def stop(self):
        self._stop.set()
        self.join(timeout=2)
        
    def run(self):
        if not INOTIFY_AVAILABLE:
            logging.warning("Inotify module not available. File watching disabled.")
            return
            
        logging.info(f"Starting file watcher for {self.backing_root}")
        logging.info(f"Maximum inotify watches: {self.max_watches}")
        
        try:
            # Setup a single recursive watcher for the root directory
            self._setup_watcher()
            
            # Monitor events
            while not self._stop.is_set():
                if self.watcher:
                    self._check_events()
                time.sleep(1)  # Small delay to prevent CPU spinning
        except Exception as e:
            logging.exception(f"Error in file watcher: {e}")
        finally:
            self._cleanup_watcher()
            
    def _setup_watcher(self):
        """Set up a single inotify watcher for the root directory and recursively watch subdirectories."""
        try:
            # Check and potentially increase system limits for inotify
            self._increase_inotify_limits()
            
            # First try to create a direct InotifyTree (efficient but can fail with many subdirectories)
            try:
                # Use a single InotifyTree for the whole tree
                self.watcher = inotify.adapters.InotifyTree(
                    self.backing_root,
                    mask=inotify.constants.IN_CLOSE_WRITE | 
                         inotify.constants.IN_MOVED_TO |
                         inotify.constants.IN_DELETE |
                         inotify.constants.IN_MOVED_FROM
                )
                if self.verbose:
                    logging.info(f"Added recursive watcher for directory tree: {self.backing_root}")
            except OSError as e:
                # If we hit "Too many open files" error (EMFILE, errno 24), we need to use fallback
                if e.errno == 24:
                    logging.warning(f"Too many open files for InotifyTree on {self.backing_root}, using fallback mode")
                    # Create a simpler Inotify watcher just for the root dir
                    self.watcher = inotify.adapters.Inotify()
                    self.watcher.add_watch(self.backing_root, 
                                        mask=inotify.constants.IN_CLOSE_WRITE | 
                                             inotify.constants.IN_MOVED_TO |
                                             inotify.constants.IN_DELETE |
                                             inotify.constants.IN_MOVED_FROM |
                                             inotify.constants.IN_CREATE |  # Need to watch for new dirs
                                             inotify.constants.IN_ISDIR)    # Flag to identify directories
                    if self.verbose:
                        logging.info(f"Added fallback watcher for root directory: {self.backing_root}")
                else:
                    # Re-raise if it's not a "too many files" error
                    raise
        except Exception as e:
            logging.error(f"Failed to set up watcher for {self.backing_root}: {e}")
            self.watcher = None
            
    def _increase_inotify_limits(self):
        """Try to increase system limits for inotify if possible."""
        try:
            # Check current max_user_watches limit
            with open('/proc/sys/fs/inotify/max_user_watches', 'r') as f:
                current = int(f.read().strip())
                
            # Only try to increase if we have root access (unlikely, but worth a try)
            if os.geteuid() == 0:  # Root user
                desired = 524288  # Common high value (512K watches)
                if current < desired:
                    try:
                        with open('/proc/sys/fs/inotify/max_user_watches', 'w') as f:
                            f.write(str(desired))
                        logging.info(f"Increased max_user_watches from {current} to {desired}")
                    except Exception as e:
                        logging.warning(f"Failed to increase max_user_watches: {e}")
            else:
                # Just log the current limit for non-root users
                logging.info(f"Current max_user_watches: {current}")
                
            # Check and log max_user_instances
            with open('/proc/sys/fs/inotify/max_user_instances', 'r') as f:
                instances = int(f.read().strip())
                logging.info(f"Current max_user_instances: {instances}")
                
        except Exception as e:
            logging.warning(f"Could not check/increase inotify limits: {e}")
    
    def _cleanup_watcher(self):
        """Clean up the watcher."""
        self.watcher = None
        if self.verbose:
            logging.debug("Cleaned up directory watcher")
    
    def _check_events(self):
        """Check for filesystem events and process them."""
        if not self.watcher:
            return
            
        try:
            for event in self.watcher.event_gen(yield_nones=False, timeout_s=0):
                _, type_names, path, filename = event
                
                if not filename:  # Skip directory events
                    continue
                    
                fullpath = os.path.join(path, filename)
                
                # We're interested in new files or modified files
                if ('IN_CLOSE_WRITE' in type_names or 'IN_MOVED_TO' in type_names):
                    # Skip directories and non-regular files
                    if not os.path.isfile(fullpath):
                        continue
                    
                    # Process the file
                    self._process_new_file(fullpath)
                
                # Handle file deletion events
                elif ('IN_DELETE' in type_names or 'IN_MOVED_FROM' in type_names):
                    # We can't check if it's a file since it's already deleted,
                    # so we'll try to handle it and let the handler deal with it
                    try:
                        relpath = os.path.relpath(fullpath, self.backing_root)
                        self._handle_deleted_file(relpath)
                    except Exception as e:
                        if self.verbose:
                            logging.warning(f"Error handling deletion of file {fullpath}: {e}")
        except Exception as e:
            logging.warning(f"Error checking events: {e}")
            # If we encounter a fatal error, recreate the watcher
            self._cleanup_watcher()
            time.sleep(5)  # Wait a bit before retrying
            self._setup_watcher()
    
    def _process_new_file(self, fullpath):
        """Process a newly created or modified file."""
        try:
            # Check if this is a file type we care about
            _, ext = os.path.splitext(fullpath.lower())
            if ext not in self.subtitle_extensions and ext not in self.video_extensions:
                return
                
            # Get relative path from backing root
            relpath = os.path.relpath(fullpath, self.backing_root)
            
            if self.verbose:
                logging.info(f"Detected new/modified file: {relpath}")
                
            # Cache the file
            self._cache_file(fullpath, relpath)
        except Exception as e:
            logging.warning(f"Error processing new file {fullpath}: {e}")

    def _handle_deleted_file(self, relpath):
        """Remove cache files for a deleted backing file."""
        # Check if this is a file type we care about
        _, ext = os.path.splitext(relpath.lower())
        if ext not in self.subtitle_extensions and ext not in self.video_extensions:
            return
            
        if self.verbose:
            logging.info(f"Detected deleted file: {relpath}")
            
        # Get cache file paths
        if ext in self.subtitle_extensions:
            # For subtitle files, we have full and meta files
            full_path = self.cache_manager.get_path_for(relpath, 'full')
            meta_path = self.cache_manager.get_path_for(relpath, 'meta')
            
            # Remove cache files if they exist
            for cache_file in [full_path, meta_path]:
                if cache_file and os.path.exists(cache_file):
                    try:
                        os.remove(cache_file)
                        # Update LRU cache state
                        with self.cache_manager.lock:
                            if cache_file in self.cache_manager.lru:
                                size, _ = self.cache_manager.lru.pop(cache_file)
                                self.cache_manager.current_cache_bytes -= size
                        if self.verbose:
                            logging.info(f"Removed cache file for deleted subtitle: {cache_file}")
                    except OSError as e:
                        logging.warning(f"Failed to remove cache file {cache_file}: {e}")
        else:
            # For video files, remove head, tail, and meta files
            head_path = self.cache_manager.get_path_for(relpath, 'head')
            tail_path = self.cache_manager.get_path_for(relpath, 'tail')
            meta_path = self.cache_manager.get_path_for(relpath, 'meta')
            
            # Remove cache files if they exist
            for cache_file in [head_path, tail_path, meta_path]:
                if cache_file and os.path.exists(cache_file):
                    try:
                        os.remove(cache_file)
                        # Update LRU cache state
                        with self.cache_manager.lock:
                            if cache_file in self.cache_manager.lru:
                                size, _ = self.cache_manager.lru.pop(cache_file)
                                self.cache_manager.current_cache_bytes -= size
                        if self.verbose:
                            logging.info(f"Removed cache file for deleted video: {cache_file}")
                    except OSError as e:
                        logging.warning(f"Failed to remove cache file {cache_file}: {e}")

    def _cache_file(self, fullpath, relpath):
        """Cache a file's head and tail (or full file for subtitles)."""
        # Check if this is a subtitle file
        _, ext = os.path.splitext(fullpath.lower())
        is_subtitle = ext in self.subtitle_extensions
        
        try:
            st = os.stat(fullpath)
            metadata = {
                'size': st.st_size, 
                'mtime': st.st_mtime,
                'atime': st.st_atime,
                'ctime': st.st_ctime,
                'mode': st.st_mode,
                'uid': st.st_uid,
                'gid': st.st_gid,
                'nlink': st.st_nlink
            }
            
            if is_subtitle:
                # For subtitle files, we fully cache the entire file
                full_path = self.cache_manager.get_path_for(relpath, 'full')
                meta_path = self.cache_manager.get_path_for(relpath, 'meta')
                
                ensure_parent_dir(meta_path)
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
                
                if self.verbose:
                    logging.info(f"Caching full subtitle file {relpath}")
                ensure_parent_dir(full_path)
                safe_copy_file(fullpath, full_path)
                self.cache_manager.add(full_path)
            else:
                # For video files, we cache head and tail portions
                head_path = self.cache_manager.get_path_for(relpath, 'head')
                tail_path = self.cache_manager.get_path_for(relpath, 'tail')
                meta_path = self.cache_manager.get_path_for(relpath, 'meta')
                
                ensure_parent_dir(meta_path)
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)
                
                # Cache head
                length = min(self.head_bytes, metadata['size'])
                if length > 0:
                    if self.verbose:
                        logging.info(f"Caching head {length} bytes for {relpath}")
                    ensure_parent_dir(head_path)
                    safe_copy_range(fullpath, head_path, length, offset=0)
                    self.cache_manager.add(head_path)
                
                # Cache tail
                if metadata['size'] <= self.tail_bytes:
                    offset = 0
                    length = metadata['size']
                else:
                    offset = metadata['size'] - self.tail_bytes
                    length = self.tail_bytes
                
                if length > 0:
                    if self.verbose:
                        logging.info(f"Caching tail {length} bytes for {relpath}")
                    ensure_parent_dir(tail_path)
                    safe_copy_range(fullpath, tail_path, length, offset=offset)
                    self.cache_manager.add(tail_path)
        except Exception as e:
            logging.exception(f"Failed to cache file {fullpath}: {e}")


# Removed CacheWorker class as we're using single-threaded processing only

def debug_startup():
    """Helper function to log basic system information for debugging purposes."""
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Current directory: {os.getcwd()}")
    logging.info(f"Current PID: {os.getpid()}")
    logging.info(f"FUSE availability: {hasattr(FUSE, '__version__')}")
    logging.info(f"Inotify availability: {INOTIFY_AVAILABLE}")
    
    # Log environment information
    try:
        import platform
        logging.info(f"Platform: {platform.platform()}")
        logging.info(f"Processor: {platform.processor()}")
        logging.info(f"CPU count: {os.cpu_count()}")
    except Exception as e:
        logging.warning(f"Error getting platform info: {e}")
        
    # Log memory info if psutil is available
    try:
        import psutil
        mem = psutil.virtual_memory()
        logging.info(f"Memory total: {mem.total / (1024**3):.2f} GB")
        logging.info(f"Memory available: {mem.available / (1024**3):.2f} GB")
    except ImportError:
        logging.info("psutil not available for memory info")
    except Exception as e:
        logging.warning(f"Error getting memory info: {e}")

def main():
    parser = argparse.ArgumentParser(description='FUSE SSD head/tail cache for Unraid')
    parser.add_argument('--config', '-c', help='Path to JSON config (optional)')
    parser.add_argument('--foreground', action='store_true', help='Run in foreground mode')
    parser.add_argument('--read-only', action='store_true', help='Mount filesystem as read-only (can be used without root)')
    parser.add_argument('--use-builtin-config', action='store_true', help='Use built-in configuration')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    # Run the debug startup function to check system information
    debug_startup()

    try:
        # Load configuration - either from file or built-in
        config = load_config(args.config, args.use_builtin_config)

        # Set logging level based on verbosity
        logging_level = logging.DEBUG if args.debug or config['GLOBAL']['VERBOSE'] else logging.INFO
        logging.basicConfig(
            level=logging_level, 
            format='%(created).6f [%(levelname)s] %(message)s'
        )

        # Always silence third-party libraries regardless of verbose setting
        logging.getLogger('inotify').setLevel(logging.WARNING)

        logging.info("Starting SSD Cache Filesystem")
        
        # Create manager with the config directly, not the path
        manager = MultiMountManager(config, foreground=args.foreground, global_read_only=args.read_only)
        
        # Setup signal handlers for proper cleanup
        import signal
        
        def signal_handler(sig, frame):
            process_name = multiprocessing.current_process().name
            process_id = os.getpid()
            logging.info(f"Received signal {sig} in process {process_name} (PID: {process_id}), shutting down...")
            
            # Only have the main process call stop_all()
            if process_name == 'MainProcess':
                logging.info(f"Main process handling shutdown")
                manager.stop_all()
            else:
                logging.info(f"Child process {process_id} exiting directly")
            
                # Child processes should just exit without calling stop_all()
            
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Clean up any stale mounts before starting
        logging.info("Checking for stale mounts before starting")
        manager._unmount_stale_points()
        
        # Add a small delay to ensure mount points are fully released by the OS
        time.sleep(1)
        
        # Mount all filesystems
        logging.info("Mounting filesystems")
        manager.mount_all()
        
    except KeyboardInterrupt:
        logging.info("Caught keyboard interrupt, shutting down...")
    except Exception as e:
        logging.exception(f"Error in main process: {e}")
    finally:
        try:
            manager.stop_all()
        except Exception as e:
            logging.exception(f"Error during cleanup: {e}")

if __name__ == '__main__':
    main()
