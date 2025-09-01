"""
UI components and utilities for YOLO Trainer
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
from typing import Callable, Optional, Any, Dict, List
from pathlib import Path
import json

from . import logger, format_timestamp

class ProgressDialog:
    """Progress dialog for long-running operations"""

    def __init__(self, parent: tk.Tk, title: str = "Processing...", message: str = "Please wait..."):
        self.parent = parent
        self.title = title
        self.message = message
        self.dialog: Optional[tk.Toplevel] = None
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value=message)
        self.cancelled = False

    def show(self) -> bool:
        """Show progress dialog and return True if not cancelled"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title(self.title)
        self.dialog.geometry("400x150")
        self.dialog.resizable(False, False)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        # Center the dialog
        self.dialog.geometry("+{}+{}".format(
            self.parent.winfo_rootx() + 50,
            self.parent.winfo_rooty() + 50
        ))

        # Progress bar
        progress_frame = ttk.Frame(self.dialog, padding="20")
        progress_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(progress_frame, textvariable=self.status_var, wraplength=350).pack(pady=(0, 10))

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            length=350
        )
        self.progress_bar.pack(pady=(0, 10))

        # Cancel button
        cancel_btn = ttk.Button(progress_frame, text="Cancel", command=self._cancel)
        cancel_btn.pack()

        self.dialog.protocol("WM_DELETE_WINDOW", self._cancel)
        self.parent.wait_window(self.dialog)

        return not self.cancelled

    def update_progress(self, value: float, status: str = ""):
        """Update progress value and status"""
        if self.dialog:
            self.progress_var.set(value)
            if status:
                self.status_var.set(status)
            self.dialog.update_idletasks()

    def _cancel(self):
        """Cancel the operation"""
        self.cancelled = True
        if self.dialog:
            self.dialog.destroy()

class ConfirmationDialog:
    """Enhanced confirmation dialog with custom options"""

    @staticmethod
    def ask_yes_no_cancel(parent: tk.Tk, title: str, message: str,
                         yes_text: str = "Yes", no_text: str = "No",
                         cancel_text: str = "Cancel") -> Optional[str]:
        """Show yes/no/cancel dialog and return user's choice"""
        dialog = tk.Toplevel(parent)
        dialog.title(title)
        dialog.geometry("350x120")
        dialog.resizable(False, False)
        dialog.transient(parent)
        dialog.grab_set()

        # Center the dialog
        dialog.geometry("+{}+{}".format(
            parent.winfo_rootx() + 50,
            parent.winfo_rooty() + 50
        ))

        result = [None]  # Use list to modify from inner function

        def set_result(value):
            result[0] = value
            dialog.destroy()

        # Message
        ttk.Label(dialog, text=message, wraplength=320).pack(pady=20, padx=20)

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        ttk.Button(button_frame, text=yes_text, command=lambda: set_result("yes")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text=no_text, command=lambda: set_result("no")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text=cancel_text, command=lambda: set_result("cancel")).pack(side=tk.RIGHT)

        parent.wait_window(dialog)
        return result[0]

class FileSelector:
    """File/directory selection utilities"""

    @staticmethod
    def select_file(parent: tk.Tk, title: str = "Select File",
                   filetypes: Optional[List[tuple]] = None,
                   initialdir: Optional[str] = None) -> Optional[str]:
        """Select a file"""
        if filetypes is None:
            filetypes = [("All files", "*.*")]

        return filedialog.askopenfilename(
            parent=parent,
            title=title,
            filetypes=filetypes,
            initialdir=initialdir
        )

    @staticmethod
    def select_directory(parent: tk.Tk, title: str = "Select Directory",
                        initialdir: Optional[str] = None) -> Optional[str]:
        """Select a directory"""
        return filedialog.askdirectory(
            parent=parent,
            title=title,
            initialdir=initialdir
        )

    @staticmethod
    def select_save_file(parent: tk.Tk, title: str = "Save File",
                        filetypes: Optional[List[tuple]] = None,
                        initialdir: Optional[str] = None,
                        defaultextension: str = "") -> Optional[str]:
        """Select file to save"""
        if filetypes is None:
            filetypes = [("All files", "*.*")]

        return filedialog.asksaveasfilename(
            parent=parent,
            title=title,
            filetypes=filetypes,
            initialdir=initialdir,
            defaultextension=defaultextension
        )

class LogViewer:
    """Log viewer widget"""

    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.frame = ttk.Frame(parent)

        # Create scrolled text widget
        self.text_widget = scrolledtext.ScrolledText(
            self.frame,
            wrap=tk.WORD,
            height=10,
            state='disabled'
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Configure tags for different log levels
        self.text_widget.tag_configure("INFO", foreground="black")
        self.text_widget.tag_configure("WARNING", foreground="orange")
        self.text_widget.tag_configure("ERROR", foreground="red")
        self.text_widget.tag_configure("SUCCESS", foreground="green")

    def add_log_entry(self, level: str, message: str, timestamp: bool = True):
        """Add a log entry"""
        if not self.text_widget.winfo_exists():
            return

        self.text_widget.config(state='normal')

        if timestamp:
            timestamp_str = format_timestamp()
            entry = f"[{timestamp_str}] {level}: {message}\n"
        else:
            entry = f"{level}: {message}\n"

        start_pos = self.text_widget.index('end-1c')
        self.text_widget.insert(tk.END, entry)

        # Apply color tag
        end_pos = self.text_widget.index('end-1c')
        self.text_widget.tag_add(level, start_pos, end_pos)

        # Auto scroll to bottom
        self.text_widget.see(tk.END)
        self.text_widget.config(state='disabled')

    def clear_logs(self):
        """Clear all log entries"""
        if self.text_widget.winfo_exists():
            self.text_widget.config(state='normal')
            self.text_widget.delete(1.0, tk.END)
            self.text_widget.config(state='disabled')

    def get_frame(self) -> ttk.Frame:
        """Get the log viewer frame"""
        return self.frame

class StatusBar:
    """Status bar widget"""

    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.frame = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)

        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar()

        # Status label
        self.status_label = ttk.Label(
            self.frame,
            textvariable=self.status_var,
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Progress bar (initially hidden)
        self.progress_bar = ttk.Progressbar(
            self.frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            length=200
        )

    def set_status(self, message: str):
        """Set status message"""
        self.status_var.set(message)
        self.parent.update_idletasks()

    def show_progress(self, show: bool = True):
        """Show or hide progress bar"""
        if show and not self.progress_bar.winfo_ismapped():
            self.progress_bar.pack(side=tk.RIGHT, padx=(0, 5))
        elif not show and self.progress_bar.winfo_ismapped():
            self.progress_bar.pack_forget()

    def update_progress(self, value: float):
        """Update progress value"""
        self.progress_var.set(value)
        self.parent.update_idletasks()

    def get_frame(self) -> ttk.Frame:
        """Get the status bar frame"""
        return self.frame

class TrainingMonitor:
    """Training progress monitor widget"""

    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.frame = ttk.LabelFrame(parent, text="Training Progress", padding="10")

        # Training info
        info_frame = ttk.Frame(self.frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(info_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.model_var = tk.StringVar()
        ttk.Label(info_frame, textvariable=self.model_var).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(info_frame, text="Epoch:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.epoch_var = tk.StringVar(value="0/0")
        ttk.Label(info_frame, textvariable=self.epoch_var).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(info_frame, text="Loss:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5))
        self.loss_var = tk.StringVar(value="0.000")
        ttk.Label(info_frame, textvariable=self.loss_var).grid(row=2, column=1, sticky=tk.W)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        # Control buttons
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill=tk.X)

        self.start_btn = ttk.Button(button_frame, text="Start Training", state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_btn = ttk.Button(button_frame, text="Stop Training", state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.pause_btn = ttk.Button(button_frame, text="Pause", state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT)

    def update_training_info(self, model: str = "", epoch: str = "", loss: str = ""):
        """Update training information"""
        if model:
            self.model_var.set(model)
        if epoch:
            self.epoch_var.set(epoch)
        if loss:
            self.loss_var.set(loss)

    def update_progress(self, value: float):
        """Update progress bar"""
        self.progress_var.set(value)
        self.parent.update_idletasks()

    def set_button_states(self, training_active: bool = False, can_start: bool = True):
        """Set button states"""
        if training_active:
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.NORMAL)
        else:
            self.start_btn.config(state=tk.NORMAL if can_start else tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.DISABLED)

    def set_start_callback(self, callback: Callable):
        """Set start training callback"""
        self.start_btn.config(command=callback)

    def set_stop_callback(self, callback: Callable):
        """Set stop training callback"""
        self.stop_btn.config(command=callback)

    def set_pause_callback(self, callback: Callable):
        """Set pause training callback"""
        self.pause_btn.config(command=callback)

    def get_frame(self) -> ttk.LabelFrame:
        """Get the training monitor frame"""
        return self.frame

# Utility functions for UI
def center_window(window: tk.Tk, width: Optional[int] = None, height: Optional[int] = None):
    """Center window on screen"""
    window.update_idletasks()

    if width and height:
        window.geometry(f"{width}x{height}")

    x = (window.winfo_screenwidth() // 2) - (window.winfo_width() // 2)
    y = (window.winfo_screenheight() // 2) - (window.winfo_height() // 2)
    window.geometry(f"+{x}+{y}")

def show_error_message(parent: tk.Tk, title: str, message: str):
    """Show error message dialog"""
    messagebox.showerror(title, message, parent=parent)

def show_info_message(parent: tk.Tk, title: str, message: str):
    """Show info message dialog"""
    messagebox.showinfo(title, message, parent=parent)

def show_warning_message(parent: tk.Tk, title: str, message: str):
    """Show warning message dialog"""
    messagebox.showwarning(title, message, parent=parent)

def ask_yes_no(parent: tk.Tk, title: str, message: str) -> bool:
    """Ask yes/no question"""
    return messagebox.askyesno(title, message, parent=parent)
