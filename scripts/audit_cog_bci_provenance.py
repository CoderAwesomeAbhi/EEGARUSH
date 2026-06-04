#!/usr/bin/env python3
"""COG-BCI (Zenodo 7413650 / concept 6874128) source provenance & construct audit.

Audit ONLY. No predictive model is run; no AUC / separability / feature importance
/ any result-dependent statistic is computed. Reads only structure + signal
headers (sampling rate, channel names, duration) needed to verify the LOCKED
one-shot protocol (`COG_BCI_ONE_SHOT_PROSPECTIVE_TEST_PROTOCOL.md`):

  calibration = eyes-open RS_Beg_EO ; scored-rest = eyes-open RS_End_EO ;
  scored-task = MATB difficult (MATBdiff) ; channels = F3,F4,F7,F8,O1,O2,T7,T8 ;
  sampling = 500 Hz.

Sources per subject: a locally-downloaded MD5-verified zip if present, otherwise a
Zenodo HTTP-range read (remotezip) of just the needed members. No COG-BCI signal
files are committed (everything lives under git-ignored data/raw/).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings("ignore")
import mne  # noqa: E402
from remotezip import RemoteZip  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "cog_bci" / "zenodo_6874128"
OUT = ROOT / "results" / "cog_bci_provenance"
OUT.mkdir(parents=True, exist_ok=True)
ZBASE = "https://zenodo.org/api/records/7413650/files/{name}/content"

SESSIONS = ["ses-S1", "ses-S2", "ses-S3"]
PRIMARY_SESSION = "ses-S1"  # predeclared single session for the one-shot test (first session)
CONDITIONS = {  # locked role -> eeg basename
    "calibration_RS_Beg_EO": "RS_Beg_EO",
    "scored_rest_RS_End_EO": "RS_End_EO",
    "scored_task_MATBdiff": "MATBdiff",
}
LOCKED_CHANNELS = ["F3", "F4", "F7", "F8", "O1", "O2", "T7", "T8"]  # T7=MAT T3, T8=MAT T4

MD5 = {
    "sub-01.zip": "23f1f74cced86b40a0a956c67bba74fb", "sub-02.zip": "f3f0b80c9752494c7e0eaba2b5cba63b",
    "sub-03.zip": "3120766d0e2d65ef0299980381f4aa3e", "sub-04.zip": "613ae1a3937261ba07ff239f80f6ebf9",
    "sub-05.zip": "9395a89068a247dc1f28bee677a2522d", "sub-06.zip": "da029812690e0e352d7b4a3f2f3231f3",
    "sub-07.zip": "1c7778dd78b514108d20c39bd13f83b0", "sub-08.zip": "3ddf5876b2ed252d7210be6bb5c54b79",
    "sub-09.zip": "84e52f812865cc500f3c8d39b82a3d5f", "sub-10.zip": "3470c8890aa6ae5bd684c2a4127554f8",
    "sub-11.zip": "c63aab36869315fa7303c76c4bfaa0e6", "sub-12.zip": "7fda5b987eb6374a257b142b839fcd28",
    "sub-13.zip": "f0b00c46f9f2c6088e4d9a8b0affea81", "sub-14.zip": "03e164e277d89070c30fa59424cd78c4",
    "sub-15.zip": "1fd226373725dc3783d33a8c62ed06d6", "sub-16.zip": "0dbab52411a3d8c2a07f7b2d749039c9",
    "sub-17.zip": "1c81adb20356f3f1429df7d0dd7bb914", "sub-18.zip": "76999fc70c7fd2d9ddbd9138deb14f8d",
    "sub-19.zip": "fe9b49a82ce7c4b0fadb2b9e7900bb65", "sub-20.zip": "6db83600139c2d075e96593d1a692ff1",
    "sub-21.zip": "34a1bb2d9f95e5c7664c0fd6c2f00333", "sub-22.zip": "4cfd73876a2de9b87a381260d750900d",
    "sub-23.zip": "e2cdcf884ca801b10951a2ede52d53cb", "sub-24.zip": "b3af9f114daa6f8a6022db580e9e012c",
    "sub-25.zip": "bbe75eb0758ff331f0807b242ee3119c", "sub-26.zip": "e650edc8edcbbb295f3fa49f0bc9c62d",
    "sub-27.zip": "0751c82f0cbeb2ee07e8ecec66d2797c", "sub-28.zip": "4d3862026ba6775f29b68a0c2b32e5f7",
    "sub-29.zip": "cfee23041f75b86b9e7d6f671c633287",
}


def md5sum(path: Path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def find_member(names, ses, base, ext):
    pat = re.compile(rf"(^|/){re.escape(ses)}/eeg/{re.escape(base)}\.{ext}$")
    for n in names:
        if pat.search(n):
            return n
    return None


def read_header_from_bytes(set_bytes, fdt_bytes, base, tmp: Path):
    (tmp / f"{base}.set").write_bytes(set_bytes)
    (tmp / f"{base}.fdt").write_bytes(fdt_bytes)
    raw = mne.io.read_raw_eeglab(str(tmp / f"{base}.set"), preload=False, verbose="ERROR")
    return float(raw.info["sfreq"]), list(raw.ch_names), raw.n_times / float(raw.info["sfreq"])


class Source:
    """Unify local zipfile and remote zip access."""
    def __init__(self, name):
        self.name = name
        self.local = SRC / name
        if self.local.exists():
            self.kind = "local"
            self.zf = zipfile.ZipFile(self.local)
        else:
            self.kind = "remote"
            self.zf = RemoteZip(ZBASE.format(name=name))

    def namelist(self):
        return self.zf.namelist()

    def read(self, member):
        return self.zf.read(member)

    def close(self):
        try:
            self.zf.close()
        except Exception:
            pass


def main():
    manifest, summary, chanver, feasibility = [], [], [], []

    for name in sorted(MD5):
        sid = name.replace(".zip", "")
        local = SRC / name
        # integrity
        if local.exists():
            got = md5sum(local)
            md5_status = "local_md5_verified" if got == MD5[name] else "local_md5_MISMATCH"
            size = local.stat().st_size
        else:
            md5_status = "remote_range_read_manifest_md5_on_record"
            size = ""
        try:
            src = Source(name)
            names = src.namelist()
        except Exception as e:  # noqa: BLE001
            manifest.append({"file": name, "source": "missing", "size_bytes": size,
                             "md5_expected": MD5[name], "md5_status": f"unreadable:{e}",
                             "n_entries": 0})
            feasibility.append({"subject_id": sid, "source": "missing", "md5_status": md5_status,
                                "calibration_present": False, "scored_rest_present": False,
                                "scored_task_present": False, "locked_design_executable": False,
                                "note": "archive unreadable"})
            continue

        manifest.append({"file": name, "source": src.kind, "size_bytes": size,
                         "md5_expected": MD5[name], "md5_status": md5_status,
                         "n_entries": len(names)})

        # per-session filename presence (suffix match -> robust to variable nesting)
        for ses in SESSIONS:
            row = {"subject_id": sid, "session": ses}
            for role, base in CONDITIONS.items():
                row[f"{role}_set"] = find_member(names, ses, base, "set") is not None
                row[f"{role}_fdt"] = find_member(names, ses, base, "fdt") is not None
            summary.append(row)

        # header verification on PRIMARY session
        ses = PRIMARY_SESSION
        primary_ok = True
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for role, base in CONDITIONS.items():
                m_set = find_member(names, ses, base, "set")
                m_fdt = find_member(names, ses, base, "fdt")
                if not m_set or not m_fdt:
                    chanver.append({"subject_id": sid, "session": ses, "role": role,
                                    "readable": False, "sfreq": "", "n_channels": "",
                                    "locked_present": 0, "missing": json.dumps(LOCKED_CHANNELS),
                                    "duration_s": ""})
                    primary_ok = False
                    continue
                try:
                    sfreq, chans, dur = read_header_from_bytes(
                        src.read(m_set), src.read(m_fdt), base, tmp)
                    present = [c for c in LOCKED_CHANNELS if c in chans]
                    missing = [c for c in LOCKED_CHANNELS if c not in chans]
                    chanver.append({"subject_id": sid, "session": ses, "role": role,
                                    "readable": True, "sfreq": sfreq, "n_channels": len(chans),
                                    "locked_present": len(present), "missing": json.dumps(missing),
                                    "duration_s": round(dur, 2)})
                    if len(present) != 8 or sfreq != 500.0:
                        primary_ok = False
                except Exception as e:  # noqa: BLE001
                    chanver.append({"subject_id": sid, "session": ses, "role": role,
                                    "readable": False, "sfreq": "", "n_channels": "",
                                    "locked_present": 0, "missing": f"read_error:{e}",
                                    "duration_s": ""})
                    primary_ok = False

        def present(ses, base):
            return (find_member(names, ses, base, "set") is not None
                    and find_member(names, ses, base, "fdt") is not None)
        feasibility.append({
            "subject_id": sid, "source": src.kind, "md5_status": md5_status,
            "primary_session": ses,
            "calibration_present": present(ses, "RS_Beg_EO"),
            "scored_rest_present": present(ses, "RS_End_EO"),
            "scored_task_present": present(ses, "MATBdiff"),
            "locked_design_executable": primary_ok,
            "note": "ok" if primary_ok else "see channel/presence rows",
        })
        src.close()
        print(f"{sid}: src={src.kind} md5={md5_status.split('_')[0]} executable={primary_ok}")

    def write(fn, rows):
        with (OUT / fn).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    write("cog_bci_source_manifest.csv", manifest)
    write("cog_bci_subject_session_condition_summary.csv", summary)
    write("cog_bci_locked_channel_verification.csv", chanver)
    write("cog_bci_protocol_feasibility.csv", feasibility)

    n_present = sum(1 for m in manifest if m["source"] != "missing")
    n_exec = sum(1 for r in feasibility if r.get("locked_design_executable"))
    print(f"\nsubjects readable: {n_present}/29 | primary-session locked-design "
          f"executable: {n_exec}/{n_present}")
    print("wrote 4 CSVs to results/cog_bci_provenance/")


if __name__ == "__main__":
    raise SystemExit(main())
