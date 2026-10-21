import glob
import os
import shutil
import subprocess
import sys
import uuid

from mcp.server.fastmcp import FastMCP

mcp = FastMCP()

MANIM_EXECUTABLE = os.getenv("MANIM_EXECUTABLE", "manim")
MANIM_QUALITY = os.getenv("MANIM_QUALITY", "-ql")
# Maximum amount of time allowed for one Manim render
MANIM_TIMEOUT = int(os.getenv("MANIM_TIMEOUT", "60"))

# This file lives in chatbot/servers_mcp/, so outputs land in chatbot/servers_mcp/media/
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
os.makedirs(BASE_DIR, exist_ok=True)

# Keep track of directories that contain a successfully rendered video
TEMP_DIRS = set()


def log(message: str) -> None:
    # stdout is reserved for the MCP protocol, so logs go to stderr
    print(message, file=sys.stderr)


def is_safe_directory(directory: str) -> bool:
    """
    Ensure a directory is located strictly inside BASE_DIR.

    This prevents the cleanup tool from deleting arbitrary
    directories outside the Manim workspace, or BASE_DIR itself.
    """
    base_path = os.path.normcase(os.path.abspath(BASE_DIR))
    target_path = os.path.normcase(os.path.abspath(directory))

    try:
        return (
            target_path != base_path
            and os.path.commonpath([base_path, target_path]) == base_path
        )
    except ValueError:
        return False


def remove_dir(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
    if os.path.exists(path):
        log(f"Warning: could not fully remove directory: {path}")


@mcp.tool()
def execute_manim_code(manim_code: str) -> str:
    """
    Execute the supplied Manim Python code and generate a video.
    """

    # Layout on success: media/manim_<id>/{scene.py, video.mp4}
    execution_id = uuid.uuid4().hex[:8]
    execution_dir = os.path.join(BASE_DIR, f"manim_{execution_id}")
    temp_dir = os.path.join(execution_dir, "temp")
    script_path = os.path.join(execution_dir, "scene.py")
    final_video_path = os.path.join(execution_dir, "video.mp4")

    os.makedirs(temp_dir, exist_ok=True)
    video_created = False

    try:
        with open(script_path, "w", encoding="utf-8") as script_file:
            script_file.write(manim_code)

        log(f"Starting Manim execution: {execution_id}")

        result = subprocess.run(
            [
                MANIM_EXECUTABLE,
                "render",
                MANIM_QUALITY,
                "--disable_caching",
                "--media_dir",
                temp_dir,
                "-o",
                "video",
                script_path,
            ],
            capture_output=True,
            text=True,
            cwd=execution_dir,
            stdin=subprocess.DEVNULL,
            timeout=MANIM_TIMEOUT,
            # Stops Python from writing __pycache__ next to scene.py
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        log(f"Manim process finished: {execution_id}")

        if result.returncode != 0:
            return (
                "Manim execution failed.\n"
                f"Return code: {result.returncode}\n\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}"
            )

        # The final video sits at temp/videos/<scene_file>/<quality>/video.mp4
        # Partial render files live deeper and are not matched by this pattern
        rendered = glob.glob(
            os.path.join(temp_dir, "videos", "*", "*", "video.mp4")
        )

        if not rendered:
            return (
                "Manim execution completed successfully, "
                "but no MP4 video file was found.\n\n"
                f"Manim output:\n{result.stdout}"
            )

        shutil.move(rendered[0], final_video_path)
        video_created = True
        TEMP_DIRS.add(execution_dir)

        return (
            "Execution successful.\n"
            f"Video generated at: {final_video_path}\n"
            f"Execution directory: {execution_dir}"
        )

    except subprocess.TimeoutExpired:
        return f"Manim execution timed out after {MANIM_TIMEOUT} seconds."

    except Exception as e:
        return (
            "Error during Manim execution:\n"
            f"{type(e).__name__}: {str(e)}"
        )

    finally:
        remove_dir(temp_dir)
        if video_created:
            # Safety net in case __pycache__ was still created
            remove_dir(os.path.join(execution_dir, "__pycache__"))
        else:
            # No video was produced, so the whole run folder (including scene.py) is junk
            remove_dir(execution_dir)


@mcp.tool()
def cleanup_manim_temp_dir(directory: str) -> str:
    """
    Delete a Manim execution directory (video, scene script, everything inside it).

    Only directories located inside the server's BASE_DIR
    are allowed to be deleted.
    """

    try:
        if not is_safe_directory(directory):
            return (
                "Cleanup rejected.\n"
                "The specified directory is outside "
                "the Manim media directory."
            )

        directory = os.path.abspath(directory)

        if not os.path.exists(directory):
            return f"Directory not found: {directory}"

        shutil.rmtree(directory)

        # Remove it from our tracking set
        TEMP_DIRS.discard(directory)

        return f"Cleanup successful for directory: {directory}"

    except Exception as e:
        return (
            f"Failed to clean up directory: {directory}\n"
            f"Error: {type(e).__name__}: {str(e)}"
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")