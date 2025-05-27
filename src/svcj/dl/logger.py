import logging
import subprocess
import threading
import time
import webbrowser
import socket
from pathlib import Path
from typing import Tuple, Optional
from torch.utils.tensorboard import SummaryWriter

def spawn_tensorboard_server(log_dir: str, initial_port: int = 6006, open_browser: bool = True, max_attempts: int = 10) -> Tuple[Optional[subprocess.Popen], int]:
    """
    Spawn a TensorBoard server in a separate process, trying multiple ports if necessary.
    
    Parameters
    ----------
    log_dir : str
        Directory containing TensorBoard logs
    initial_port : int, default=6006
        Initial port to try for TensorBoard
    open_browser : bool, default=True
        Whether to automatically open browser to TensorBoard URL
    max_attempts : int, default=10
        Maximum number of port attempts before giving up
        
    Returns
    -------
    Tuple[Optional[subprocess.Popen], int]
        The TensorBoard server process (or None if failed) and the port used.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    current_port = initial_port

    for attempt in range(max_attempts):
        port_to_try = initial_port + attempt
        cmd = ["tensorboard", "--logdir", log_dir, "--port", str(port_to_try), "--host", "0.0.0.0"]
        
        # Check if port is likely available before launching TensorBoard
        # This is a quick check, but TensorBoard might still fail for other reasons
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port_to_try)) == 0:
                print(f"TensorBoard: Port {port_to_try} is in use, trying next...")
                continue # Port is occupied

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            time.sleep(3)  # Increased sleep time for TensorBoard to initialize

            if process.poll() is None:  # Process is running
                print(f"✓ TensorBoard server started on http://localhost:{port_to_try}")
                print(f"  Log directory: {log_dir}")
                current_port = port_to_try

                if open_browser:
                    def open_browser_delayed():
                        time.sleep(2) # Further delay for server readiness
                        try:
                            webbrowser.open(f"http://localhost:{current_port}")
                            print(f"  Browser opened automatically for port {current_port}")
                        except Exception as e:
                            print(f"  Could not open browser for port {current_port} automatically: {e}")
                    
                    browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
                    browser_thread.start()
                return process, current_port
            else:
                # Process failed to start, get error output
                stdout, stderr = process.communicate(timeout=5) # Added timeout
                error_msg = stderr if stderr else stdout
                print(f"TensorBoard: Failed to start on port {port_to_try}. Error: {error_msg.strip()}")
                if "Address already in use" in error_msg or "EADDRINUSE" in error_msg:
                    continue # Try next port
                # For other errors, don't retry immediately, but the loop will continue for max_attempts

        except FileNotFoundError:
            raise RuntimeError(
                "TensorBoard not found. Please install it with: pip install tensorboard"
            )
        except subprocess.TimeoutExpired:
             print(f"TensorBoard: Timeout communicating with process on port {port_to_try}. Assuming failure.")
             if process and process.poll() is None: # if process exists and is running, kill it
                process.kill()
                process.communicate()
        except Exception as e:
            # Catch other potential errors during Popen or communication
            print(f"TensorBoard: Error launching on port {port_to_try}: {e}")
            # Continue to next attempt if it's a general error, specific errors might be handled above
    
    print(f"⚠ TensorBoard: Failed to start on ports {initial_port}-{initial_port + max_attempts -1}. Please check logs or start manually.")
    return None, initial_port # Return None and initial_port if all attempts fail

class TBLogger:
    def __init__(self, log_dir: str, log_to_file: bool = False, log_file: str = "train.log", 
                 log_level: int = logging.INFO, spawn_tensorboard: bool = True, 
                 tensorboard_port: int = 6006, open_browser: bool = True):
        """
        TensorBoard Logger with optional automatic TensorBoard server spawning.
        
        Parameters
        ----------
        log_dir : str
            Directory for TensorBoard logs
        log_to_file : bool, default=False
            Whether to log to file
        log_file : str, default="train.log"
            Log file name
        log_level : int, default=logging.INFO
            Logging level
        spawn_tensorboard : bool, default=True
            Whether to automatically spawn TensorBoard server
        tensorboard_port : int, default=6006
            Preferred initial port for TensorBoard server
        open_browser : bool, default=True
            Whether to automatically open browser to TensorBoard
        """
        self.log_dir = log_dir
        self.tensorboard_process: Optional[subprocess.Popen] = None
        self.actual_tensorboard_port: int = tensorboard_port # Store the actual port used
        
        # TensorBoard writer
        self.w = SummaryWriter(log_dir)

        # Python logger
        self.logger = logging.getLogger("TBLogger")
        self.logger.setLevel(log_level)
        self.logger.propagate = False  # Avoid duplicate logs

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Optional file handler
        if log_to_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        
        # Spawn TensorBoard server if requested
        if spawn_tensorboard:
            try:
                self.tensorboard_process, self.actual_tensorboard_port = spawn_tensorboard_server(
                    log_dir, initial_port=tensorboard_port, open_browser=open_browser
                )
                if self.tensorboard_process:
                    self.logger.info(f"TensorBoard server spawned on port {self.actual_tensorboard_port}")
                else:
                    self.logger.warning(f"Failed to spawn TensorBoard server after multiple attempts. Initial port was {tensorboard_port}.")
            except Exception as e:
                self.logger.warning(f"Failed to spawn TensorBoard server (initial port {tensorboard_port}): {e}")
                self.tensorboard_process = None

    def log(self, step: int, value: float, tag: str = "train/loss", verbose: bool = True):
        # TensorBoard log
        self.w.add_scalar(tag, value, step)
        # Standard log
        if verbose:
            self.logger.info(f"{tag} @ step {step}: {value:.6f}")

    def log_msg(self, msg: str, level: int = logging.INFO):
        # General-purpose message logging
        self.logger.log(level, msg)

    def close(self):
        self.w.close()
        
        # Terminate TensorBoard server if it was spawned
        if self.tensorboard_process is not None:
            try:
                self.tensorboard_process.terminate()
                self.tensorboard_process.wait(timeout=5)  # Wait up to 5 seconds
                self.logger.info("TensorBoard server terminated")
            except subprocess.TimeoutExpired:
                self.tensorboard_process.kill()
                self.logger.warning("TensorBoard server killed (timeout)")
            except Exception as e:
                self.logger.warning(f"Error terminating TensorBoard server: {e}")
        
        # Close all handlers (especially important if logging to file)
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
