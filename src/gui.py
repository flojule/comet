#!/usr/bin/env python3
# gui.py
"""Desktop front end: pick a recording, pick a camera, run SAM 3.

    python src/gui.py

Deliberately thin.  Every non-trivial step — reading bags, listing topics,
running the model, measuring orientation — lives in a module with tests
(mcap_source, media, stairs_pipeline, stair_orientation).  This file only
collects arguments, runs the work on a background thread, and prints what
comes back, so the part that cannot be tested without a display is the part
where the least can go wrong.

Needs Tk, which some Python builds omit:  sudo apt install python3-tk
"""
from __future__ import annotations

import queue
import sys
import threading
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as e:                        # pragma: no cover - env dependent
    sys.exit(
        f"This GUI needs Tk, which is missing from this Python ({e}).\n"
        "  Debian/Ubuntu:  sudo apt install python3-tk\n"
        "  macOS (brew):   brew install python-tk\n"
        "Or use the command line instead:\n"
        "  python src/stairs_pipeline.py --list <folder>\n"
        "  python src/stairs_pipeline.py <folder> --topic <topic> --prompt stairs"
    )

APP_TITLE = "Comet — segment & measure"
PROMPT_SUGGESTIONS = ["stairs", "staircase", "steps", "door", "ramp", "person"]


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(APP_TITLE)
        root.geometry("860x640")
        root.minsize(720, 520)

        self.source = tk.StringVar()
        self.topic = tk.StringVar()
        self.prompt = tk.StringVar(value="stairs")
        self.out = tk.StringVar(value="output/stairs")
        self.max_frames = tk.StringVar(value="")
        self.checkpoint = tk.StringVar(value="")
        self.do_orientation = tk.BooleanVar(value=True)

        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None

        self._build()
        self.root.after(100, self._drain)

    # -- layout ---------------------------------------------------------------
    def _build(self) -> None:
        pad = dict(padx=8, pady=4)
        frm = ttk.Frame(self.root)
        frm.pack(fill="x", **pad)
        frm.columnconfigure(1, weight=1)

        r = 0
        ttk.Label(frm, text="Recording").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.source).grid(row=r, column=1, sticky="ew")
        btns = ttk.Frame(frm)
        btns.grid(row=r, column=2, sticky="e")
        ttk.Button(btns, text="Folder…", command=self._pick_folder).pack(side="left")
        ttk.Button(btns, text="File…", command=self._pick_file).pack(side="left")

        r += 1
        ttk.Label(frm, text="Camera topic").grid(row=r, column=0, sticky="w")
        self.topic_box = ttk.Combobox(frm, textvariable=self.topic, state="readonly")
        self.topic_box.grid(row=r, column=1, sticky="ew")
        ttk.Button(frm, text="Scan bag",
                   command=self._scan).grid(row=r, column=2, sticky="e")

        r += 1
        ttk.Label(frm, text="Find (text prompt)").grid(row=r, column=0, sticky="w")
        ttk.Combobox(frm, textvariable=self.prompt,
                     values=PROMPT_SUGGESTIONS).grid(row=r, column=1, sticky="ew")

        r += 1
        ttk.Label(frm, text="Output stem").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.out).grid(row=r, column=1, sticky="ew")

        r += 1
        opts = ttk.Frame(frm)
        opts.grid(row=r, column=1, sticky="w")
        ttk.Label(opts, text="Max frames").pack(side="left")
        ttk.Entry(opts, textvariable=self.max_frames, width=8).pack(side="left", padx=(4, 12))
        ttk.Checkbutton(opts, text="Measure orientation",
                        variable=self.do_orientation).pack(side="left")

        r += 1
        ttk.Label(frm, text="Checkpoint").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.checkpoint).grid(row=r, column=1, sticky="ew")
        ttk.Label(frm, text="(blank = auto)").grid(row=r, column=2, sticky="w")

        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=8, pady=(8, 4))
        self.run_btn = ttk.Button(bar, text="Run", command=self._run)
        self.run_btn.pack(side="left")
        ttk.Button(bar, text="Preflight", command=self._preflight).pack(side="left", padx=6)
        self.status = ttk.Label(bar, text="Ready")
        self.status.pack(side="left", padx=12)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.progress.pack(side="right")

        self.log = tk.Text(self.root, wrap="word", height=20)
        self.log.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        sb = ttk.Scrollbar(self.log, command=self.log.yview)
        sb.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=sb.set)
        self._say("Pick a recording folder, press Scan bag, choose a topic, then Run.")
        self._say("A plain video, a photo, or a folder of photos works too — "
                  "no topic needed for those.")

    # -- helpers --------------------------------------------------------------
    def _say(self, text: str) -> None:
        self._queue.put(("log", str(text)))

    def _drain(self) -> None:
        """Pump worker messages onto the Tk thread; Tk is not thread-safe."""
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "log":
                    self.log.insert("end", payload + "\n")
                    self.log.see("end")
                elif kind == "status":
                    self.status.config(text=payload)
                elif kind == "topics":
                    self.topic_box["values"] = payload
                    if payload and not self.topic.get():
                        self.topic.set(payload[0])
                elif kind == "done":
                    self.progress.stop()
                    self.run_btn.config(state="normal")
                    self.status.config(text=payload)
                elif kind == "error":
                    self.progress.stop()
                    self.run_btn.config(state="normal")
                    self.status.config(text="Failed")
                    messagebox.showerror(APP_TITLE, payload)
        except queue.Empty:
            pass
        self.root.after(100, self._drain)

    def _busy(self, what: str) -> bool:
        if self._worker and self._worker.is_alive():
            messagebox.showinfo(APP_TITLE, "Already running — wait for it to finish.")
            return True
        self.run_btn.config(state="disabled")
        self.status.config(text=what)
        self.progress.start(12)
        return False

    def _spawn(self, fn) -> None:
        def wrapped():
            try:
                fn()
            except Exception as e:              # noqa: BLE001 - surfaced in UI
                self._queue.put(("log", traceback.format_exc()))
                self._queue.put(("error", f"{type(e).__name__}: {e}"))
        self._worker = threading.Thread(target=wrapped, daemon=True)
        self._worker.start()

    # -- actions --------------------------------------------------------------
    def _pick_folder(self) -> None:
        d = filedialog.askdirectory(title="Recording folder (mcap bags or photos)")
        if d:
            self.source.set(d)
            self._scan()

    def _pick_file(self) -> None:
        f = filedialog.askopenfilename(
            title="Video, photo or .mcap",
            filetypes=[("All supported", "*.mp4 *.mov *.mkv *.avi *.webm "
                                         "*.jpg *.jpeg *.png *.mcap"),
                       ("All files", "*.*")])
        if f:
            self.source.set(f)
            self._scan()

    def _scan(self) -> None:
        src = self.source.get().strip()
        if not src:
            return
        if self._busy("Scanning…"):
            return

        def work():
            from media import classify
            kind = classify(src)
            self._say(f"Input looks like: {kind}")
            if kind == "mcap":
                from mcap_source import list_image_topics
                topics = list_image_topics(src)
                self._say(f"Found {len(topics)} image topic(s):")
                for t in topics:
                    self._say("   " + t.describe())
                names = [t.topic for t in topics]
                # Colour first — depth and infra are rarely what you want to
                # run a text prompt against.
                names.sort(key=lambda n: (("color" not in n and "rgb" not in n), n))
                self._queue.put(("topics", names))
                self._queue.put(("done", f"{len(names)} topic(s)"))
            else:
                self._queue.put(("topics", []))
                self._queue.put(("done", f"{kind} — no topic needed"))
        self._spawn(work)

    def _preflight(self) -> None:
        if self._busy("Preflight…"):
            return

        def work():
            import io
            import contextlib
            import sam3_preflight
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = sam3_preflight.main()
            for line in buf.getvalue().splitlines():
                self._say(line)
            self._queue.put(("done", "Preflight OK" if code == 0 else "Preflight failed"))
        self._spawn(work)

    def _run(self) -> None:
        src = self.source.get().strip()
        if not src:
            messagebox.showwarning(APP_TITLE, "Pick a recording first.")
            return
        prompt = self.prompt.get().strip()
        if not prompt:
            messagebox.showwarning(APP_TITLE, "Enter what to look for, e.g. 'stairs'.")
            return
        try:
            max_frames = int(self.max_frames.get()) if self.max_frames.get().strip() else None
        except ValueError:
            messagebox.showwarning(APP_TITLE, "Max frames must be a whole number.")
            return
        if self._busy("Running…"):
            return

        topic = self.topic.get().strip() or None
        out = self.out.get().strip() or "output/stairs"
        ckpt = self.checkpoint.get().strip() or None
        want_ori = self.do_orientation.get()

        def work():
            import stairs_pipeline
            res = stairs_pipeline.run(
                src, prompt=prompt, topic=topic, out_stem=out,
                max_frames=max_frames, checkpoint=ckpt,
                orientation=want_ori, log=self._say,
                progress=lambda n: self._queue.put(("status", f"{n} frames…")),
            )
            self._say("")
            self._say(f"Done — {res.frames} frames, {len(res.objects)} object(s)")
            if res.overlay:
                self._say(f"Overlay video: {res.overlay}")
            if res.orientation.get("angle_deg_mean") is not None:
                self._say(f"Mean angle: {res.orientation['angle_deg_mean']:.1f}deg")
            self._queue.put(("done", "Finished"))
        self._spawn(work)


def main() -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
