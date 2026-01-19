#!/usr/bin/env python3
"""
OCTA SLICER - Octatrack-Style Sample Slicer
Simulates the 128x64 pixel LCD display of the Elektron Octatrack.

Keyboard Controls:
- Left/Right arrows: Move start/end position of current slice
- Up/Down arrows: Switch between slices
- Space: Toggle between editing start/end of current slice
- Enter: Process and export slices
- L: Load file
- B: Batch load
- O: Output directory
- S: Cycle slice count
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import threading
import subprocess
import os
import wave
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Optional dependencies
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


# ============================================================================
# 128x64 PIXEL LCD SIMULATION
# ============================================================================

# Display dimensions (like real Octatrack)
LCD_WIDTH = 128
LCD_HEIGHT = 64
PIXEL_SCALE = 6  # Each LCD pixel = 6x6 screen pixels

# Colors - Clean monochrome
LCD_OFF = "#000000"      # Pixel off (black)
LCD_ON = "#ffffff"       # Pixel on (white)
LCD_BG = "#000000"       # Screen background


# ============================================================================
# BITMAP FONT - 5x7 pixel characters (fits ~21 chars across, ~9 lines)
# ============================================================================

FONT_5X7 = {
    'A': ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    'B': ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    'C': ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    'D': ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    'E': ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    'F': ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    'G': ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    'H': ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    'I': ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    'J': ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    'K': ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    'L': ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    'M': ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    'N': ["10001", "10001", "11001", "10101", "10011", "10001", "10001"],
    'O': ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    'P': ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    'Q': ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    'R': ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    'S': ["01110", "10001", "10000", "01110", "00001", "10001", "01110"],
    'T': ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    'U': ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    'V': ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    'W': ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    'X': ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    'Y': ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    'Z': ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    '0': ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    '1': ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    '2': ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
    '3': ["01110", "10001", "00001", "00110", "00001", "10001", "01110"],
    '4': ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    '5': ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    '6': ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    '7': ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    '8': ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    '9': ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    ' ': ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    '.': ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ':': ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    '-': ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    '/': ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    '[': ["01110", "01000", "01000", "01000", "01000", "01000", "01110"],
    ']': ["01110", "00010", "00010", "00010", "00010", "00010", "01110"],
    '<': ["00010", "00100", "01000", "10000", "01000", "00100", "00010"],
    '>': ["01000", "00100", "00010", "00001", "00010", "00100", "01000"],
    '=': ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
    '+': ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    '*': ["00000", "10101", "01110", "11111", "01110", "10101", "00000"],
    '#': ["01010", "01010", "11111", "01010", "11111", "01010", "01010"],
    '%': ["11001", "11010", "00010", "00100", "01000", "01011", "10011"],
    '_': ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    '|': ["00100", "00100", "00100", "00100", "00100", "00100", "00100"],
    '(': ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ')': ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    '?': ["01110", "10001", "00001", "00110", "00100", "00000", "00100"],
    '!': ["00100", "00100", "00100", "00100", "00100", "00000", "00100"],
    ',': ["00000", "00000", "00000", "00000", "00110", "00100", "01000"],
    "'": ["00100", "00100", "00000", "00000", "00000", "00000", "00000"],
}

# Smaller 3x5 font for compact display
FONT_3X5 = {
    'A': ["010", "101", "111", "101", "101"],
    'B': ["110", "101", "110", "101", "110"],
    'C': ["011", "100", "100", "100", "011"],
    'D': ["110", "101", "101", "101", "110"],
    'E': ["111", "100", "110", "100", "111"],
    'F': ["111", "100", "110", "100", "100"],
    'G': ["011", "100", "101", "101", "011"],
    'H': ["101", "101", "111", "101", "101"],
    'I': ["111", "010", "010", "010", "111"],
    'J': ["001", "001", "001", "101", "010"],
    'K': ["101", "110", "100", "110", "101"],
    'L': ["100", "100", "100", "100", "111"],
    'M': ["101", "111", "101", "101", "101"],
    'N': ["101", "111", "111", "111", "101"],
    'O': ["010", "101", "101", "101", "010"],
    'P': ["110", "101", "110", "100", "100"],
    'Q': ["010", "101", "101", "110", "011"],
    'R': ["110", "101", "110", "101", "101"],
    'S': ["011", "100", "010", "001", "110"],
    'T': ["111", "010", "010", "010", "010"],
    'U': ["101", "101", "101", "101", "010"],
    'V': ["101", "101", "101", "010", "010"],
    'W': ["101", "101", "111", "111", "101"],
    'X': ["101", "101", "010", "101", "101"],
    'Y': ["101", "101", "010", "010", "010"],
    'Z': ["111", "001", "010", "100", "111"],
    '0': ["010", "101", "101", "101", "010"],
    '1': ["010", "110", "010", "010", "111"],
    '2': ["110", "001", "010", "100", "111"],
    '3': ["110", "001", "010", "001", "110"],
    '4': ["101", "101", "111", "001", "001"],
    '5': ["111", "100", "110", "001", "110"],
    '6': ["011", "100", "110", "101", "010"],
    '7': ["111", "001", "010", "010", "010"],
    '8': ["010", "101", "010", "101", "010"],
    '9': ["010", "101", "011", "001", "110"],
    ' ': ["000", "000", "000", "000", "000"],
    '.': ["000", "000", "000", "000", "010"],
    ':': ["000", "010", "000", "010", "000"],
    '-': ["000", "000", "111", "000", "000"],
    '/': ["001", "001", "010", "100", "100"],
    '<': ["001", "010", "100", "010", "001"],
    '>': ["100", "010", "001", "010", "100"],
    '[': ["110", "100", "100", "100", "110"],
    ']': ["011", "001", "001", "001", "011"],
}


# ============================================================================
# AUDIO PROCESSING
# ============================================================================

@dataclass
class AudioData:
    samples: np.ndarray  # Mono samples for display
    samples_original: np.ndarray  # Original samples (may be stereo)
    sample_rate: int
    channels: int
    duration: float
    filepath: str
    filename: str


@dataclass
class SlicePoint:
    start: int
    end: int
    index: int


class AudioProcessor:
    OCTA_SAMPLE_RATE = 44100
    OCTA_BIT_DEPTH = 16
    OCTA_MAX_VALUE = 32767

    @staticmethod
    def load_audio(filepath: str) -> Optional[AudioData]:
        try:
            if HAS_SOUNDFILE:
                samples, sample_rate = sf.read(filepath, dtype='float32')
            else:
                samples, sample_rate = AudioProcessor._load_wav_native(filepath)

            # Store original samples
            samples_original = samples.copy()

            if len(samples.shape) > 1:
                channels = samples.shape[1]
                samples_mono = np.mean(samples, axis=1)
            else:
                channels = 1
                samples_mono = samples
                # Make original 2D for consistency if mono
                samples_original = samples

            duration = len(samples_mono) / sample_rate
            filename = os.path.basename(filepath)

            return AudioData(
                samples=samples_mono,
                samples_original=samples_original,
                sample_rate=sample_rate,
                channels=channels,
                duration=duration,
                filepath=filepath,
                filename=filename
            )
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None

    @staticmethod
    def _load_wav_native(filepath: str) -> Tuple[np.ndarray, int]:
        with wave.open(filepath, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            raw_data = wav_file.readframes(n_frames)

            if sample_width == 2:
                dtype = np.int16
                max_val = 32767
            else:
                dtype = np.int16
                max_val = 32767

            samples = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
            samples = samples / max_val

            if n_channels > 1:
                samples = samples.reshape(-1, n_channels)

            return samples, sample_rate

    @staticmethod
    def resample(samples: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
        if orig_rate == target_rate:
            return samples

        num_samples = int(len(samples) * target_rate / orig_rate)

        if HAS_SCIPY:
            # scipy.signal.resample works with multi-dimensional arrays along axis 0
            return signal.resample(samples, num_samples, axis=0).astype(np.float32)
        else:
            # Fallback interpolation - handle both mono and stereo
            indices = np.linspace(0, len(samples) - 1, num_samples)
            if len(samples.shape) > 1:
                # Stereo - interpolate each channel
                resampled = np.zeros((num_samples, samples.shape[1]), dtype=np.float32)
                for ch in range(samples.shape[1]):
                    resampled[:, ch] = np.interp(indices, np.arange(len(samples)), samples[:, ch])
                return resampled
            else:
                return np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)

    @staticmethod
    def validate_and_convert(samples: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int, List[str]]:
        warnings = []
        if sample_rate != AudioProcessor.OCTA_SAMPLE_RATE:
            warnings.append(f"RESAMPLED {sample_rate}>{AudioProcessor.OCTA_SAMPLE_RATE}")
            samples = AudioProcessor.resample(samples, sample_rate, AudioProcessor.OCTA_SAMPLE_RATE)
            sample_rate = AudioProcessor.OCTA_SAMPLE_RATE

        max_val = np.max(np.abs(samples))
        if max_val > 1.0:
            warnings.append("NORMALIZED")
            samples = samples / max_val

        return samples, sample_rate, warnings

    @staticmethod
    def export_slice(samples: np.ndarray, sample_rate: int, filepath: str, stereo: bool = False) -> Tuple[bool, str]:
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Determine number of channels
            if stereo and len(samples.shape) > 1:
                n_channels = samples.shape[1]
                samples_out = samples
            else:
                n_channels = 1
                # Convert to mono if needed
                if len(samples.shape) > 1:
                    samples_out = np.mean(samples, axis=1)
                else:
                    samples_out = samples

            samples_int = np.clip(samples_out * AudioProcessor.OCTA_MAX_VALUE,
                                  -AudioProcessor.OCTA_MAX_VALUE,
                                  AudioProcessor.OCTA_MAX_VALUE).astype(np.int16)

            if HAS_SOUNDFILE:
                sf.write(filepath, samples_int, sample_rate, subtype='PCM_16')
            else:
                with wave.open(filepath, 'wb') as wav_file:
                    wav_file.setnchannels(n_channels)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(samples_int.tobytes())

            return AudioProcessor.validate_wav(filepath, stereo)
        except Exception as e:
            return False, str(e)

    @staticmethod
    def validate_wav(filepath: str, stereo: bool = False) -> Tuple[bool, str]:
        try:
            with wave.open(filepath, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()
                n_channels = wav_file.getnchannels()
                n_frames = wav_file.getnframes()

                if sample_rate != 44100:
                    return False, f"BAD RATE {sample_rate}"
                if sample_width != 2:
                    return False, f"BAD BITS {sample_width*8}"
                if n_frames == 0:
                    return False, "EMPTY FILE"
                if stereo and n_channels != 2:
                    return False, "NOT STEREO"

                wav_file.readframes(min(1000, n_frames))
                return True, "OK"
        except Exception as e:
            return False, str(e)


# ============================================================================
# LCD DISPLAY WIDGET - 128x64 PIXEL SIMULATION
# ============================================================================

class LCDDisplay(tk.Canvas):
    """Simulates a 128x64 pixel backlit LCD like the Octatrack."""

    def __init__(self, parent, **kwargs):
        # Calculate canvas size
        canvas_width = LCD_WIDTH * PIXEL_SCALE
        canvas_height = LCD_HEIGHT * PIXEL_SCALE

        kwargs['width'] = canvas_width
        kwargs['height'] = canvas_height
        kwargs['bg'] = LCD_BG
        kwargs['highlightthickness'] = 0
        kwargs['borderwidth'] = 0

        super().__init__(parent, **kwargs)

        # Pixel buffer (128x64)
        self.pixels = np.zeros((LCD_HEIGHT, LCD_WIDTH), dtype=np.uint8)

        # Pre-create pixel rectangles for performance (no gaps)
        self.pixel_rects = {}
        for y in range(LCD_HEIGHT):
            for x in range(LCD_WIDTH):
                x1 = x * PIXEL_SCALE
                y1 = y * PIXEL_SCALE
                x2 = x1 + PIXEL_SCALE
                y2 = y1 + PIXEL_SCALE
                rect = self.create_rectangle(x1, y1, x2, y2, fill=LCD_OFF, outline='', width=0)
                self.pixel_rects[(x, y)] = rect

    def clear(self):
        """Clear the display."""
        self.pixels.fill(0)

    def set_pixel(self, x: int, y: int, on: bool = True):
        """Set a single pixel."""
        if 0 <= x < LCD_WIDTH and 0 <= y < LCD_HEIGHT:
            self.pixels[y, x] = 1 if on else 0

    def draw_text(self, x: int, y: int, text: str, small: bool = False):
        """Draw text at pixel position using bitmap font."""
        font = FONT_3X5 if small else FONT_5X7
        char_width = 4 if small else 6
        char_height = 5 if small else 7

        cursor_x = x
        for char in text.upper():
            if char in font:
                bitmap = font[char]
                for row_idx, row in enumerate(bitmap):
                    for col_idx, pixel in enumerate(row):
                        if pixel == '1':
                            px = cursor_x + col_idx
                            py = y + row_idx
                            self.set_pixel(px, py, True)
            cursor_x += char_width

    def draw_hline(self, x: int, y: int, length: int):
        """Draw horizontal line."""
        for i in range(length):
            self.set_pixel(x + i, y, True)

    def draw_vline(self, x: int, y: int, length: int):
        """Draw vertical line."""
        for i in range(length):
            self.set_pixel(x, y + i, True)

    def draw_rect(self, x: int, y: int, w: int, h: int, filled: bool = False):
        """Draw rectangle."""
        if filled:
            for py in range(y, y + h):
                for px in range(x, x + w):
                    self.set_pixel(px, py, True)
        else:
            self.draw_hline(x, y, w)
            self.draw_hline(x, y + h - 1, w)
            self.draw_vline(x, y, h)
            self.draw_vline(x + w - 1, y, h)

    def draw_waveform(self, samples: np.ndarray, x: int, y: int, width: int, height: int,
                      slice_start: float = 0.0, slice_end: float = 1.0,
                      current_slice_start: int = -1, current_slice_end: int = -1):
        """Draw waveform in specified area."""
        if len(samples) == 0:
            return

        center_y = y + height // 2
        samples_per_col = max(1, len(samples) // width)

        for col in range(width):
            sample_idx = col * samples_per_col
            chunk = samples[sample_idx:sample_idx + samples_per_col]
            if len(chunk) == 0:
                continue

            # Get amplitude
            max_amp = np.max(np.abs(chunk))
            bar_height = int(max_amp * (height // 2 - 1))

            # Check if in current slice region
            col_ratio = col / width
            in_slice = slice_start <= col_ratio <= slice_end

            if bar_height > 0:
                # Draw symmetric waveform
                for dy in range(-bar_height, bar_height + 1):
                    if in_slice:
                        self.set_pixel(x + col, center_y + dy, True)
                    elif col % 2 == 0:  # Dimmed effect outside slice
                        self.set_pixel(x + col, center_y + dy, True)

        # Draw slice markers
        if current_slice_start >= 0:
            marker_x = x + int(current_slice_start / len(samples) * width)
            if 0 <= marker_x - x < width:
                for dy in range(height):
                    self.set_pixel(marker_x, y + dy, True)

        if current_slice_end >= 0:
            marker_x = x + int(current_slice_end / len(samples) * width)
            if 0 <= marker_x - x < width:
                for dy in range(height):
                    if (y + dy) % 2 == 0:  # Dashed line for end
                        self.set_pixel(marker_x, y + dy, True)

    def draw_progress_bar(self, x: int, y: int, width: int, progress: float):
        """Draw a progress bar."""
        self.draw_rect(x, y, width, 5, filled=False)
        fill_width = int((width - 2) * progress)
        if fill_width > 0:
            self.draw_rect(x + 1, y + 1, fill_width, 3, filled=True)

    def refresh(self):
        """Update the canvas to reflect pixel buffer."""
        for y in range(LCD_HEIGHT):
            for x in range(LCD_WIDTH):
                rect = self.pixel_rects[(x, y)]
                color = LCD_ON if self.pixels[y, x] else LCD_OFF
                self.itemconfig(rect, fill=color)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class OctaSlicer(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("OCTA SLICER")
        self.configure(bg='#000000')
        self.resizable(False, False)

        # Center window on screen
        self.update_idletasks()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = LCD_WIDTH * PIXEL_SCALE
        win_h = LCD_HEIGHT * PIXEL_SCALE
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.geometry(f"{win_w}x{win_h}+{x}+{y}")

        # State
        self.audio_data: Optional[AudioData] = None
        self.output_dir: str = os.path.expanduser("~/Desktop/OctaSlices")
        self.num_slices: int = 16
        self.slices: List[SlicePoint] = []
        self.current_slice_idx: int = 0
        self.editing_start: bool = True
        self.move_amount: int = 100
        self.status_msg: str = "READY"
        self.mode: str = "MAIN"  # MAIN, LOAD, BATCH, PROCESS
        self.stereo_mode: bool = False  # False = mono, True = stereo

        # Batch queue
        self.file_queue: List[str] = []
        self.current_file_idx: int = -1
        self.processing: bool = False
        self.process_progress: float = 0.0

        # Create LCD display (no padding - borderless)
        self.lcd = LCDDisplay(self)
        self.lcd.pack(padx=0, pady=0)

        # Make LCD focusable and grab focus on click
        self.lcd.configure(takefocus=True)
        self.lcd.bind('<Button-1>', lambda e: self._grab_focus())
        self.bind('<Button-1>', lambda e: self._grab_focus())

        # Bind all keys globally using bind_all (fires once)
        # +/= to move position right, -/_ to move position left
        self.bind_all('<equal>', lambda e: self._move_position(self.move_amount))
        self.bind_all('<plus>', lambda e: self._move_position(self.move_amount))
        self.bind_all('<minus>', lambda e: self._move_position(-self.move_amount))
        self.bind_all('<underscore>', lambda e: self._move_position(-self.move_amount))
        self.bind_all('<Shift-equal>', lambda e: self._move_position(self.move_amount * 10))
        self.bind_all('<Shift-minus>', lambda e: self._move_position(-self.move_amount * 10))
        # Arrow keys for slice selection and start/end toggle
        self.bind_all('<Up>', lambda e: self._change_slice(-1))
        self.bind_all('<Down>', lambda e: self._change_slice(1))
        self.bind_all('<Left>', lambda e: self._toggle_edit_mode())
        self.bind_all('<Right>', lambda e: self._toggle_edit_mode())
        # Space to play current slice
        self.bind_all('<space>', lambda e: self._play_slice())
        self.bind_all('<Return>', lambda e: self._process())
        self.bind_all('<l>', lambda e: self._load_file())
        self.bind_all('<L>', lambda e: self._load_file())
        self.bind_all('<b>', lambda e: self._batch_load())
        self.bind_all('<B>', lambda e: self._batch_load())
        self.bind_all('<o>', lambda e: self._select_output())
        self.bind_all('<O>', lambda e: self._select_output())
        self.bind_all('<s>', lambda e: self._cycle_slices())
        self.bind_all('<S>', lambda e: self._cycle_slices())
        self.bind_all('<m>', lambda e: self._toggle_stereo())
        self.bind_all('<M>', lambda e: self._toggle_stereo())
        self.bind_all('<q>', lambda e: self.destroy())
        self.bind_all('<Q>', lambda e: self.destroy())
        self.bind_all('<Escape>', lambda e: self.destroy())

        # Initial render
        self._render()

        # Grab focus after window is shown
        self.after(100, self._grab_focus)

    def _grab_focus(self):
        """Focus this window."""
        self.lift()
        self.focus_force()

    def _render(self):
        """Render the entire display."""
        self.lcd.clear()

        # Header bar
        self.lcd.draw_hline(0, 8, 128)
        self.lcd.draw_text(2, 1, "OCTA SLICER", small=True)

        # Show stereo/mono mode
        mode_str = "ST" if self.stereo_mode else "MO"
        self.lcd.draw_text(75, 1, mode_str, small=True)

        # Show slice count on right
        self.lcd.draw_text(100, 1, f"S:{self.num_slices:02d}", small=True)

        if self.mode == "MAIN":
            self._render_main()
        elif self.mode == "PROCESS":
            self._render_processing()

        # Status bar at bottom
        self.lcd.draw_hline(0, 55, 128)
        self.lcd.draw_text(2, 57, self.status_msg[:21], small=True)

        # Key hints
        if not self.processing:
            self.lcd.draw_text(90, 57, "L:LD", small=True)

        self.lcd.refresh()

    def _render_main(self):
        """Render main editing view."""
        if self.audio_data:
            # File info
            name = self.audio_data.filename[:18]
            self.lcd.draw_text(2, 11, name, small=True)

            # Duration
            secs = self.audio_data.duration
            mins = int(secs // 60)
            secs = secs % 60
            self.lcd.draw_text(90, 11, f"{mins}:{secs:04.1f}", small=True)

            # Waveform area (main display)
            wave_y = 18
            wave_h = 24

            # Draw border
            self.lcd.draw_rect(0, wave_y, 128, wave_h)

            # Calculate current slice region for highlighting
            slice_start = 0.0
            slice_end = 1.0
            cs_start = -1
            cs_end = -1

            if self.slices and self.current_slice_idx < len(self.slices):
                current = self.slices[self.current_slice_idx]
                total = len(self.audio_data.samples)
                slice_start = current.start / total
                slice_end = current.end / total
                cs_start = current.start
                cs_end = current.end

            # Draw waveform
            self.lcd.draw_waveform(
                self.audio_data.samples,
                2, wave_y + 2, 124, wave_h - 4,
                slice_start, slice_end,
                cs_start, cs_end
            )

            # Slice info panel
            if self.slices:
                current = self.slices[self.current_slice_idx]

                # Slice number
                self.lcd.draw_text(2, 44, f"SL:{self.current_slice_idx+1:02d}/{len(self.slices):02d}", small=True)

                # Edit mode indicator
                mode_str = "START" if self.editing_start else "END"
                self.lcd.draw_text(45, 44, f"[{mode_str}]", small=True)

                # Position
                pos = current.start if self.editing_start else current.end
                pos_sec = pos / self.audio_data.sample_rate
                self.lcd.draw_text(90, 44, f"{pos_sec:.2f}S", small=True)

                # Visual slice indicator bar
                self.lcd.draw_rect(2, 51, 124, 3)
                for i, sl in enumerate(self.slices):
                    x = 2 + int((sl.start / len(self.audio_data.samples)) * 122)
                    if i == self.current_slice_idx:
                        self.lcd.draw_vline(x, 49, 5)
                    else:
                        self.lcd.set_pixel(x, 52, True)
        else:
            # No file loaded
            self.lcd.draw_text(20, 25, "NO FILE LOADED", small=False)
            self.lcd.draw_text(15, 40, "PRESS L TO LOAD", small=True)
            self.lcd.draw_text(15, 47, "PRESS B FOR BATCH", small=True)

    def _render_processing(self):
        """Render processing view."""
        self.lcd.draw_text(30, 20, "PROCESSING", small=False)

        if self.audio_data:
            name = self.audio_data.filename[:20]
            self.lcd.draw_text(2, 32, name, small=True)

        # Progress bar
        self.lcd.draw_progress_bar(10, 42, 108, self.process_progress)

        pct = int(self.process_progress * 100)
        self.lcd.draw_text(55, 50, f"{pct}%", small=True)

    def _move_position(self, delta: int):
        if not self.audio_data or not self.slices or self.processing:
            return

        current = self.slices[self.current_slice_idx]
        total = len(self.audio_data.samples)

        if self.editing_start:
            new_pos = max(0, min(current.end - 100, current.start + delta))
            if self.current_slice_idx > 0:
                prev_end = self.slices[self.current_slice_idx - 1].end
                new_pos = max(prev_end, new_pos)
            current.start = new_pos
        else:
            new_pos = max(current.start + 100, min(total, current.end + delta))
            if self.current_slice_idx < len(self.slices) - 1:
                next_start = self.slices[self.current_slice_idx + 1].start
                new_pos = min(next_start, new_pos)
            current.end = new_pos

        self._render()

    def _change_slice(self, delta: int):
        if not self.slices or self.processing:
            return

        new_idx = self.current_slice_idx + delta
        if 0 <= new_idx < len(self.slices):
            self.current_slice_idx = new_idx
            self.editing_start = True
            self._render()

    def _toggle_edit_mode(self):
        if not self.processing:
            self.editing_start = not self.editing_start
            self._render()

    def _play_slice(self):
        """Play the current slice audio."""
        if not self.audio_data or not self.slices or self.processing:
            return

        current = self.slices[self.current_slice_idx]
        slice_samples = self.audio_data.samples[current.start:current.end]

        if len(slice_samples) == 0:
            return

        self.status_msg = f"PLAYING SL:{self.current_slice_idx + 1:02d}"
        self._render()

        # Play audio in background thread
        def play_audio():
            try:
                if HAS_SOUNDDEVICE:
                    # Use sounddevice for playback
                    sd.play(slice_samples, self.audio_data.sample_rate)
                    sd.wait()
                else:
                    # Fallback: save temp file and use afplay (macOS)
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                        temp_path = f.name

                    # Convert to 16-bit and save
                    samples_int = np.clip(slice_samples * 32767, -32767, 32767).astype(np.int16)
                    with wave.open(temp_path, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(self.audio_data.sample_rate)
                        wav_file.writeframes(samples_int.tobytes())

                    # Play with afplay (macOS)
                    subprocess.run(['afplay', temp_path], capture_output=True)
                    os.unlink(temp_path)

                self.after(0, self._playback_done)
            except Exception as e:
                self.after(0, lambda: self._playback_error(str(e)))

        thread = threading.Thread(target=play_audio, daemon=True)
        thread.start()

    def _playback_done(self):
        self.status_msg = "READY"
        self._render()

    def _playback_error(self, error: str):
        self.status_msg = f"PLAY ERR: {error[:12]}"
        self._render()

    def _toggle_stereo(self):
        if self.processing:
            return
        self.stereo_mode = not self.stereo_mode
        self.status_msg = "STEREO MODE" if self.stereo_mode else "MONO MODE"
        self._render()

    def _cycle_slices(self):
        if self.processing:
            return

        options = [4, 8, 16, 32, 64]
        try:
            idx = options.index(self.num_slices)
            self.num_slices = options[(idx + 1) % len(options)]
        except ValueError:
            self.num_slices = 16

        if self.audio_data:
            self._create_slices()

        self.status_msg = f"SLICES: {self.num_slices}"
        self._render()

    def _create_slices(self):
        if not self.audio_data:
            return

        total = len(self.audio_data.samples)
        slice_size = total // self.num_slices

        self.slices = []
        for i in range(self.num_slices):
            start = i * slice_size
            end = start + slice_size if i < self.num_slices - 1 else total
            self.slices.append(SlicePoint(start=start, end=end, index=i))

        self.current_slice_idx = 0
        self.editing_start = True

    def _load_file(self):
        if self.processing:
            return

        filetypes = [
            ("Audio files", "*.wav *.aif *.aiff *.mp3 *.flac *.ogg"),
            ("WAV files", "*.wav"),
            ("All files", "*.*")
        ]

        filepath = filedialog.askopenfilename(title="LOAD AUDIO", filetypes=filetypes)
        if filepath:
            self._load_audio(filepath)

    def _load_audio(self, filepath: str):
        self.status_msg = "LOADING..."
        self._render()

        audio = AudioProcessor.load_audio(filepath)
        if audio:
            self.audio_data = audio
            self._create_slices()
            self.status_msg = f"LOADED {audio.sample_rate}HZ"
        else:
            self.status_msg = "LOAD ERROR"
            self.audio_data = None

        self.mode = "MAIN"
        self._render()

    def _batch_load(self):
        if self.processing:
            return

        filetypes = [
            ("Audio files", "*.wav *.aif *.aiff *.mp3 *.flac"),
            ("All files", "*.*")
        ]

        filepaths = filedialog.askopenfilenames(title="BATCH LOAD", filetypes=filetypes)
        if filepaths:
            self.file_queue = list(filepaths)
            self.current_file_idx = 0
            self.status_msg = f"QUEUE: {len(filepaths)} FILES"
            self._load_audio(filepaths[0])

    def _select_output(self):
        if self.processing:
            return

        directory = filedialog.askdirectory(title="OUTPUT DIR", initialdir=self.output_dir)
        if directory:
            self.output_dir = directory
            self.status_msg = "OUTPUT SET"
            self._render()

    def _process(self):
        if not self.audio_data or not self.slices or self.processing:
            return

        self.processing = True
        self.mode = "PROCESS"
        self.process_progress = 0.0
        self._render()

        thread = threading.Thread(target=self._do_process)
        thread.daemon = True
        thread.start()

    def _do_process(self):
        try:
            base_name = os.path.splitext(self.audio_data.filename)[0]

            # Use original samples for stereo, mono samples otherwise
            if self.stereo_mode and self.audio_data.channels > 1:
                source_samples = self.audio_data.samples_original
            else:
                source_samples = self.audio_data.samples

            samples, rate, warnings = AudioProcessor.validate_and_convert(
                source_samples,
                self.audio_data.sample_rate
            )

            total = len(self.slices)
            errors = []

            for i, slice_pt in enumerate(self.slices):
                scale = rate / self.audio_data.sample_rate
                start = int(slice_pt.start * scale)
                end = int(slice_pt.end * scale)

                slice_samples = samples[start:end]

                filename = f"{base_name}-{i+1}.wav"
                filepath = os.path.join(self.output_dir, filename)

                success, msg = AudioProcessor.export_slice(slice_samples, rate, filepath, self.stereo_mode)
                if not success:
                    errors.append(f"S{i+1}: {msg}")

                self.process_progress = (i + 1) / total
                self.after(0, self._render)

            # Done
            if errors:
                self.after(0, lambda: self._process_done(f"DONE W/ERRORS"))
            else:
                self.after(0, lambda: self._process_done("EXPORT COMPLETE"))

        except Exception as e:
            self.after(0, lambda: self._process_done(f"ERROR: {str(e)[:15]}"))

    def _process_done(self, msg: str):
        self.processing = False
        self.mode = "MAIN"
        self.status_msg = msg
        self._render()

        # Auto-advance batch queue
        if self.file_queue and self.current_file_idx < len(self.file_queue) - 1:
            self.current_file_idx += 1
            self._load_audio(self.file_queue[self.current_file_idx])


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    if not HAS_SOUNDFILE:
        print("NOTE: soundfile not installed - WAV only")
    if not HAS_SCIPY:
        print("NOTE: scipy not installed - basic resampling")

    app = OctaSlicer()
    app.mainloop()


if __name__ == "__main__":
    main()
