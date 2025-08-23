# VidCacheFS - Fuse File System for Media Files

This project was created to solve the annoyance of waiting 5–10 seconds for playback to start on Plex due to hard drives being spundown  by default to save power in my Unraid setup. It is inspired by the Video Preloader script for RAM caching. Since it may be useful for others, I am sharing it here.

This is provided as-is without any warranty—it works well for my setup, but your experience may vary. Only tested it on Unraid with Plex. Feel free to modify it to suit your needs or share improvements!

## What Is It?

VidCacheFS is a FUSE-based file system built to eliminate playback latency in media applications on systems where hard drives are spun down when not in use, by caching critical portions of video files on SSD storage for seamless, instant access.

Tested on Unraid with Plex and installable through the user script plugin. Requires the Python plugin, and the Python packages 'fusepy' and 'inotify' (install via `pip install fusepy inotify`).

This script can enable caching for any directory by mounting it through VidCacheFS, allowing seamless access to cached media files.

## How It Works

VidCacheFS is designed to minimize drive spinup and optimize playback performance by intelligently managing caching:

- Directory listings are cached to avoid unnecessary drive spinup and reduce playback latency.
- File attributes are cached so metadata requests do not wake the backing drive, further improving responsiveness.
- Configurable head and tail regions of each file are cached; any read request within these regions is served instantly from SSD.
- The backing drive is proactively spun up before the cache is depleted, ensuring uninterrupted access.
- Subtitle files are fully cached to eliminate playback delays while waiting for the backing drive to become available.

## Features

- **Multiple Independently Configurable Mounts:** Supports several mount points, each with its own cache and configuration, allowing tailored caching strategies for different media libraries.
- **Periodic Configurable Scan:** Automatically scans directories at user-defined intervals to update the cache. New or modified files are promptly cached, and stale cache files are deleted.
- **System Event-Driven Caching:** Listens for filesystem events (such as new file additions) to immediately cache newly added media, minimizing playback delays.
- **Read-Only and Write Modes:** Operates in read-only mode for safe media access, or passthrough write mode to allow for example deletion of media files on the backing drive directly through applications like Plex.
- **Flexible Caching Policies:** Head/tail region sizes and cache limits are fully configurable per mount.

## Requirements
- Unraid
  - Python plugin
  - User scripts plugin
- Python 3.x
- fusepy (`pip install fusepy`)
- inotify (`pip install inotify`)
- Sufficient SSD storage for caching

## Configuration

Configuration is provided as a JSON object, either embedded in the script or in a separate `config.json` file. Each mount point is defined as an entry in the `MOUNTS` array, allowing you to specify multiple independent cache setups:

```javascript
{
  "GLOBAL": {
    "VERBOSE": true,
    "MAX_CACHE_SIZE_BYTES": 1850 * 1024 * 1024 * 1024,
    "READ_ONLY": false,
    "CHECK_FILE_MODIFICATIONS": false
  },
  "MOUNTS": [
    {
      "BACKING_DIR": "/mnt/user/video/Dolby Atmos Demo",
      "MOUNT_POINT": "/mnt/ssd_cache/AtmosDemo",
      "CACHE_DIR": "/mnt/cache/ssd_cache/AtmosDemo",
      "MAX_FILES": 20,
      "HEAD_BYTES": 65 * 1024 * 1024,
      "TAIL_BYTES": 1 * 1024 * 1024,
      "SCHEDULES": [
        {
          "path": "/mnt/user/video/Dolby Atmos Demo",
          "scan_interval_seconds": 24 * 3600,
          "pattern": "*.{mkv,mp4,avi,mov,m4v,mpg,mpeg,wmv,flv,webm}"
        }
      ],
      "OPEN_SPINUP_TTL": 5
    },
    // Add more mount objects as needed
  ]
}
```

**Option Descriptions:**
- `GLOBAL`: Global settings for all mounts.
  - `VERBOSE`: Enables detailed logging.
  - `MAX_CACHE_SIZE_BYTES`: Total available cache size on the disk/folder. This is shared across all mounts.
  - `READ_ONLY`: If true, disables cache writes.
  - `CHECK_FILE_MODIFICATIONS`: Enables periodic file modification checks if inotify is unavailable. Could cause drive spinup at scheduled interval. Should not be used when inotify is available.
- `MOUNTS`: Array of mount configurations. Each object defines a separate cache/mount.
  - `BACKING_DIR`: Directory on HDD to mirror.
  - `MOUNT_POINT`: FUSE mount location.
  - `CACHE_DIR`: SSD cache storage location.
  - `MAX_FILES`: Maximum number of files to cache for this mount. Once this limit is reached, the least recently used files will be evicted from the cache to make room for new ones. The system will calculate the total space needed based on this number and the head/tail sizes.
  - `HEAD_BYTES` / `TAIL_BYTES`: Number of bytes to cache from start/end of files.
  - `SCHEDULES`: Pre-warm cache schedules (scan interval, path, file pattern).
  - `OPEN_SPINUP_TTL`: Duration (in seconds) to keep the backing file open on first cache access, ensuring the drive spins up and is ready for subsequent reads.

You can define as many mount points as needed, each with its own cache policy and schedule.

## Installation on Unraid

1. Install the 'Python 3 for UNRAID' community application.

2. Go to Settings > Python 3 and enter the following commands into the auto-execution script editor to ensure dependencies are re-installed after reboot:
   ```bash
   pip install fusepy
   pip install inotify
   ```

3. Install the 'User scripts' community application.

4. Under Settings > User script add a new script called 'Mount SSD Cache'. Copy and paste the contents of vidCacheFS.py into the script editor.

5. Edit the configuration section at the top of the script to match your needs based on the options described above.

6. Set schedule of the 'Mount SSD Cache' to 'At Startup of Array'.

7. Create another script called 'Unmount SSD Cache'. Copy and paste the contents of the code below into the script editor. Set schedule to 'At Stop of Array'.
   ```bash
   #!/bin/bash
   #Simple script to terminate SSD Cache FUSE process

   # Find all Python processes that match our target
   PIDs=$(ps aux | grep -E "Mount SSD Cache" | grep -v grep | awk '{print $2}')

   if [ -n "$PIDs" ]; then
       echo "Found SSD Cache processes: $PIDs"
       
       # Send SIGINT to each process
       for pid in $PIDs; do
           echo "Sending SIGINT to process $pid"
           kill -2 "$pid"
       done
       
       echo "SIGINT signals sent. Filesystem will unmount gracefully."
   else
       echo "No SSD Cache processes found running"
   fi
   ```

8. Press 'Run in background' on the mount script to start the fuse filesystem. It should now begin caching. You can open the log of the script to validate it is running.

## Important Considerations

- Docker containers that use the cache share must be started only after the filesystem has been mounted. Consider issuing a delayed start for these containers to ensure the filesystem is fully available before they attempt to access cached files.
 
- If the filesystem is restarted or remounted, any Docker containers using the cache must also be restarted, otherwise the cache mount inside the container won't populate.

## Plex Configuration

Edit the Plex docker container to add a new host path, mapping the root folder of the cache mount to `/media-cache`.
Set the container to delayed start to ensure it is started after the file system is mounted (array start), otherwise you will find `/media-cache` empty.

To make Plex use the new cache mount point:
1. Edit your libraries and add a new path pointing to the `/media-cache` folder.
2. Do not remove the original path yet, as doing so will cause Plex to lose all metadata and treat all media as new.
3. After the Plex libraries have finished updating, play the cached version of a video file to check if VidCacheFS is functioning properly.
4. When satisfied that everything works, you can delete the original path to the media files from the library configuration.

Disable auto emptying of the trash in Plex and if possible periodic scanning. If for whatever reason file-mount fails the media-cache folder within Plex will look empty you don't want your libraries to be updated to empty.

## Command Line Arguments

When running VidCacheFS directly from the command line instead of through the User Scripts plugin, it supports the following arguments:

| Argument                      | Description                                                                                   |
|-------------------------------|-----------------------------------------------------------------------------------------------|
| `--config <path>`, `-c <path>`| Path to the configuration JSON file (optional).                                                |
| `--foreground`                | Run the FUSE filesystem in the foreground for debugging purposes. Only mounts the first mount point listed.                             |
| `--read-only`                 | Mount the filesystem as read-only (can be used without root).                                 |
| `--use-builtin-config`        | Use the built-in configuration instead of an external config file.                             |

Example usage:
```bash
python fuse_ssd_cache_head_tail.py --config /path/to/config.json --foreground --read-only
```
